import numpy as np
import torch
import pandas as pd
import time
import os
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.sampling import get_polytope_samples
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch import gen_candidates_torch
from batch_reactor_LeuDH import Reactor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""To optimize another reactor these things have to be changed: reactor import (line21), name in save_checkpoint (line68), name in excel save (line269+276)))"""
#=================================================================================================================================================================
#Reactor Simulation
#=================================================================================================================================================================
reactor = Reactor()
params = reactor.params

#=======================================================================================================================================
#Optimization
#=======================================================================================================================================
#Results saving=================================================================================================================================================================
def save_checkpoint(X, Y, hv_history, t_iter, reactor, params, filename):
    Y_obj = torch.tensor(Y[:, :3], dtype=torch.double)
    Y_con = torch.tensor(Y[:, 3], dtype=torch.double)

    feasible_mask = (Y_con >= params["X_target"])
    if feasible_mask.sum() > 0:
        mask = np.zeros(len(Y), dtype=bool)
        feasible_idx = feasible_mask.nonzero(as_tuple=True)[0]
        local_pareto = is_non_dominated(Y_obj[feasible_mask])
        mask[feasible_idx[local_pareto].numpy()] = True
    else:
        mask = is_non_dominated(Y_obj).numpy()

    pareto_X = 10 ** X[mask]
    pareto_Y = Y[mask, :3]
    pareto_PPO_final = np.array(reactor.ppo_final_history)[mask].reshape(-1, 1)

    col_x = list(reactor.v_names) + ["PPO_final [mM]"]
    col_y = ["STY", "TON_E", "TON_COF"]
    df = pd.DataFrame(np.hstack([pareto_X, pareto_PPO_final, pareto_Y]), columns=col_x + col_y)

    df_hv = pd.DataFrame({
        "Iteration": range(1, len(hv_history) + 1),
        "Hypervolume": hv_history,
        "Duration [s]": t_iter,
    })

    with pd.ExcelWriter(filename) as writer:
        df.to_excel(writer, sheet_name="Pareto", index=False)
        df_hv.to_excel(writer, sheet_name="Hypervolume", index=False)

#State of the loop ssaving==========================================================================================================================
STATE_FILE = "Batch_LeuDH_paretofront_2008.npz"

def save_state(X, Y, hv_history, t_iter, reactor, it, filename=STATE_FILE, max_retries=5, retry_delay=1.0):
    base, ext = os.path.splitext(filename)
    tmp_filename = f"{base}.tmp{ext}"
    np.savez(tmp_filename, X=X, Y=Y, hv_history=np.array(hv_history), t_iter=np.array(t_iter), ppo_final_history=np.array(reactor.ppo_final_history), it=it)

    for attempt in range(max_retries):
        try:
            os.replace(tmp_filename, filename)
            return
        except PermissionError:
            if attempt == max_retries - 1:
                print(f"Warning: {filename} could not be updated after {max_retries} attempts, skipping this checkpoint.")
                return
            time.sleep(retry_delay)


#boundaries of the parameter domain in v/v (PPO, NAD, GluDH, FDH)=======================================================================
LOWER = np.asarray(reactor.LOWER, dtype=float)
UPPER = np.asarray(reactor.UPPER, dtype=float)

w =np.asarray(reactor.weights, dtype=float) #weighting of initial volumes to account for the volume of AF stock solution

#Initial design or start with old data==========================================================================================================================
N_INIT = 100                   #number of initial evaluations
d = len(LOWER)                 #dimensions of parameter domain

bounds_real = torch.stack([torch.tensor(LOWER, dtype=torch.double), torch.tensor(UPPER, dtype=torch.double),])   #boundaries for sampling

vol_idx = np.nonzero(w)[0]
inequality_constraints = [(torch.tensor(vol_idx, dtype=torch.long),-torch.tensor(w[vol_idx], dtype=torch.double),-1.0)]

V_max = 1.0
V_max_safe = V_max - 0.05

w_torch = torch.tensor(w, dtype=torch.double, device=device)
lower_torch = torch.tensor(LOWER, dtype=torch.double, device=device)

def project_log10(Z):
    L, wt = lower_torch.to(Z), w_torch.to(Z)
    diff = torch.pow(10.0, Z) - L
    slack = V_max_safe - (wt * L).sum()
    denom = (wt * diff).sum(dim=-1, keepdim=True).clamp_min(1e-30)
    t = (slack / denom).clamp(max=1.0)
    return torch.log10(L + t * diff)

class VolumeProjectedAcqf(AcquisitionFunction):
    def __init__(self, acq_function):
        super().__init__(model=acq_function.model)
        self.acq_function = acq_function
    def forward(self, X):
        return self.acq_function(project_log10(X))
    
NUM_RESTARTS, RAW_SAMPLES, IC_POOL_SIZE = 10, 512, 4096

ic_pool = torch.log10(get_polytope_samples(
    n=IC_POOL_SIZE, bounds=bounds_real,
    inequality_constraints=inequality_constraints,
    n_burnin=10000, n_thinning=32,
)).to(device)

def make_batch_initial_conditions(acqf, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES):
    idx = torch.randperm(ic_pool.shape[0], device=device)[:raw_samples]
    Z = ic_pool[idx].unsqueeze(1)
    with torch.no_grad():
        vals = torch.cat([acqf(Z[i:i + 128]) for i in range(0, Z.shape[0], 128)])
    return Z[vals.topk(min(num_restarts, Z.shape[0])).indices]


if os.path.exists(STATE_FILE):
    with np.load(STATE_FILE) as data:
        X = data["X"]
        Y = data["Y"]
        hv_history = data["hv_history"].tolist()
        t_iter = data["t_iter"].tolist()
        reactor.ppo_final_history = data["ppo_final_history"].tolist()
        it = int(data["it"])
    print(f"Resuming from checkpoint at iteration {it}")
else:
    X_real = get_polytope_samples(
        n=N_INIT,
        bounds=bounds_real,
        inequality_constraints=inequality_constraints,
        n_burnin=10000,
        n_thinning=32,
    ).numpy()
    X = np.log10(X_real)
    Y = np.array([reactor.simulate(row) for row in X_real])
    hv_history = []
    t_iter = []
    it = 0

print(len(X), len(Y), len(hv_history), len(reactor.ppo_final_history))
X_TARGET_WARPED = -np.log(1 - params["X_target"])   #fixed warped conversion constraint for the GP model

#Termination criteria=================================================================================================================================
# #Criterium 1:
#if the hypervolume for hv_window iterations doesnt improve by at least hv_tol*100 % 
ref_point_hv = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double)
hv_tol = 0.00005
hv_window = 400

#Criterium 2:
#if there are min_pareto_points points on the determined pareto frontier
min_pareto_points = 50

#Criterium 3:
#maximum amount of iterations
max_bo = 5000

#Bayesian Optimization loop=====================================================================================================================================================
prev_state = None
while it < max_bo:
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    #training data
    train_x = torch.tensor(X, dtype=torch.double)
    train_y = torch.tensor(Y, dtype=torch.double)

    #warping the conversion constraint to ensure that the GP model can handle the constraint properly
    g_warped = -torch.log(torch.clamp(1 - train_y[:, 3], min=1e-12))
    train_y = torch.cat([train_y[:, :3], g_warped.unsqueeze(-1)], dim=-1)

    train_y_objectives = train_y[:, :3]    #STY, TON_E, TON_COF
    train_y_constraint = train_y[:, 3:4]   #g = -log(1 - X_PPO)
        
    
    #boundaries transformed into log10
    bounds = torch.tensor(np.array([np.log10(LOWER), np.log10(UPPER)]),dtype=torch.double)

    #combined Gaussian Process------------------------------------------------------------------------------------------------------------------------------------------------------------------
    model = SingleTaskGP(
        train_x,
        train_y,
        input_transform=Normalize(d=d, bounds=bounds),
        outcome_transform=Standardize(m=4),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    if prev_state is not None:
        hypers = {k: v for k, v in prev_state.items()
                if "outcome_transform" not in k and "input_transform" not in k}
        model.load_state_dict(hypers, strict=False)

    tgp = time.perf_counter()
    fit_gpytorch_mll(mll)
    print(f"Time GP fitting:{time.perf_counter() - tgp}s")

    prev_state = model.state_dict()

    #ensuring conversion constraint while determining points of pareto frontier
    feasible_mask = (train_y_constraint[:, 0] >= X_TARGET_WARPED)

    if feasible_mask.sum() > 0:
        pareto_mask_full = torch.zeros(len(train_y), dtype=torch.bool)
        feasible_idx = feasible_mask.nonzero(as_tuple=True)[0]
        local_pareto = is_non_dominated(train_y_objectives[feasible_mask])
        pareto_mask_full[feasible_idx[local_pareto]] = True
    else:
        pareto_mask_full = is_non_dominated(train_y_objectives)

    pareto_Y_obj = train_y_objectives[pareto_mask_full]
    n_pareto = int(pareto_mask_full.sum())

    #hypervolume calculation
    ref_point_acq = infer_reference_point(pareto_Y_obj).to(device)
    hv = Hypervolume(ref_point_hv)
    current_hv = hv.compute(pareto_Y_obj)                                       #current hypervolume of pareto points
    hv_history.append(float(current_hv))                                        #history for stopping criterium 1

    #GP model transformed to GPU for AF-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    model_gpu = model.to(device)
    bounds_gpu = bounds.to(device)

    #Acquisition function-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    constraints = [lambda Z: X_TARGET_WARPED - Z[..., 3]]                                                    #conversion constraint: X_target - X <=0

    #qEHVI
    tehvi = time.perf_counter()
    partitioning = FastNondominatedPartitioning(
        ref_point=ref_point_acq,
        Y=pareto_Y_obj.to(device),
    )
    qehvi = qLogExpectedHypervolumeImprovement(
        model=model_gpu,
        ref_point=ref_point_acq,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([64])),
        partitioning=partitioning,
        constraints=constraints,
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1, 2]),
    )
    print(f"Time qEHVI construction: {time.perf_counter() - tehvi}s")

    torch.cuda.synchronize()
    taf = time.perf_counter()
    qehvi_proj = VolumeProjectedAcqf(qehvi)
    batch_ics  = make_batch_initial_conditions(qehvi_proj)
    candidate, _ = optimize_acqf(
        acq_function=qehvi_proj,
        bounds=bounds_gpu,
        q=1,
        num_restarts=NUM_RESTARTS,
        batch_initial_conditions=batch_ics,
        gen_candidates=gen_candidates_torch,
        options={"stopping_criterion_options": {"maxiter": 200}},
    )
    candidate = project_log10(candidate)

    torch.cuda.synchronize()
    print(f"Time AF optimization: {time.perf_counter() - taf}s")

    candidates = candidate.detach().cpu().numpy()

    for x_new in candidates:
        y_new = reactor.simulate(10**x_new)

        X = np.vstack([X, x_new])
        Y = np.vstack([Y, y_new])

    torch.cuda.synchronize()
    t_iter.append(time.perf_counter() - t0)

    #Termination criteria---------------------------------------------------------------------------------------------------------------------------------------
    if len(hv_history) > hv_window:
        
        hv_old_1 = hv_history[-2]               #hypervolume before current iteration
        hv_old_20 = hv_history[-hv_window-1]    #hyperolume hv_window iterations ago
        hv_new = hv_history[-1]                 #current hypervolume

        rel_improvement_1 = (hv_new - hv_old_1) / abs(hv_old_1)     #relative improvement of hypervolume since last iteration
        rel_improvement_20 = ((hv_new - hv_old_20) / abs(hv_old_20))/hv_window  #median relative improvement of hypervolume over hv_window iterations

        print(f"Median HV improvement over {hv_window} iterations: {100*rel_improvement_20:.10f}%")
        print(f"HV improvement since last iteration: {100*rel_improvement_1:.10f}%")

        if rel_improvement_20 < hv_tol and n_pareto >= min_pareto_points:
            print(
                "Stopping criterion reached."
            )
            save_checkpoint(X, Y, hv_history, t_iter, reactor, params, "Batch_LeuDH_paretofront_2008.xlsx")
            save_state(X, Y, hv_history, t_iter, reactor, it + 1)
            break

    checkpoint_interval = 5   #save every __ iterations

    if (it + 1) % checkpoint_interval == 0:
        save_checkpoint(X, Y, hv_history, t_iter, reactor, params, "Batch_LeuDH_paretofront_2008.xlsx")
        save_state(X, Y, hv_history, t_iter, reactor, it + 1)



    print(f"Time Iteration: {time.perf_counter() - t0}s")
    print(f"Iteration {it+1} complete")
    print(f"Pareto-Punkte: {n_pareto}")
    print("=====================================================================================")
    it += 1
