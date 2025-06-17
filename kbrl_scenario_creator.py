# ------------ KBRL Learner initialization values ------------------
from kbrl_control import KBRL_Control, Learner
from algorithms.kernel import GaussianKernel
from algorithms.projectron import SVvariable, Projectron


# ----------------- scenario parameters ------------------------

scenario_1 = { 'n_prbs': 80, 'n_embb': 3, 'n_mmtc': 0}
scenario_2 = { 'n_prbs': 150, 'n_embb': 3, 'n_mmtc': 2}
scenario_3 = { 'n_prbs': 100, 'n_embb': 1, 'n_mmtc': 4}
scenario_4 = { 'n_prbs': 70,  'n_embb': 1, 'n_mmtc': 1}
scenarios = [scenario_1, scenario_2, scenario_3, scenario_4]

state_variables_embb = ['cbr_traffic', 'cbr_th', 'cbr_prb', 'cbr_queue', 'cbr_snr']
state_variables_mmtc = ['devices', 'avg_rep', 'delay']

alfa = 0.05 # learning parameter

# initial offset and initial action are initialized at random
embb_sec = (2, 8)
embb_a = (2, 4)
mmtc_sec = (1, 4)
mmtc_a = (2, 10)

# -------------------- create KBRL agent -------------------------

def create_kbrl_agent(rng, n, accuracy_range = [0.99, 0.999]):
    '''
    Returns kbrl agent:
    - rng: for random number generation
    - n: selects the scenario (0, 1, 2)
    - accuracy_range: for the learner
    - budget: number of support vectors in memory
    '''
    sc = scenarios[n]
    n_prbs = sc['n_prbs']
    n_embb = sc['n_embb']
    n_mmtc = sc['n_mmtc']
    embb_dim = len(state_variables_embb)
    mmtc_dim = len(state_variables_mmtc)

    learners = [] 
    i = 0

    # create one learner instance per slice
    for _ in range(n_embb):
        sv = SVvariable() # create support vector memory
        kernel = GaussianKernel(sv,1) # kernel
        algorithm = Projectron(kernel) # online classifier
        initial_action = rng.integers(embb_a[0], embb_a[1])
        sec = rng.integers(embb_sec[0], embb_sec[1])
        learner = Learner(algorithm, slice(i,i+embb_dim), initial_action, sec)
        learners.append(learner)
        i += embb_dim

    for _ in range(n_mmtc):
        sv = SVvariable()
        kernel = GaussianKernel(sv,1)
        algorithm = Projectron(kernel)
        initial_action = rng.integers(mmtc_a[0], mmtc_a[1])
        sec = rng.integers(mmtc_sec[0], mmtc_sec[1])
        learner = Learner(algorithm, slice(i,i+mmtc_dim), initial_action, sec)
        learners.append(learner)
        i += mmtc_dim

    kbrl_agent = KBRL_Control(learners, n_prbs, alfa = alfa, accuracy_range = accuracy_range)

    return kbrl_agent
