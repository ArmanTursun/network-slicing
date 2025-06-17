class QR_Learner:
    '''
    Auxiliary data class to hold the elements for a single SLA constraint learner.
    '''
    def __init__(self, algorithm, indexes, initial_action, sla_threshold, constraint_type, kpi_key, kpi_index):
        self.algorithm = algorithm
        self.indexes = indexes
        self.initial_action = initial_action
        self.sla_threshold = sla_threshold
        self.constraint_type = constraint_type # 'upper' (e.g., delay) or 'lower' (e.g., throughput)
        self.kpi_key = kpi_key # Key to find the KPI in the environment's info dictionary
        self.kpi_index = kpi_index



class MultiObjectiveAgentController:
    """Manages multiple smart SVGPAgents to make safe and performant decisions."""
    def __init__(self, learners, n_prbs, exploration_factor=1.0, resource_cost_factor=0.1):
        # 'learners' is now a dictionary of smart SVGPAgents
        self.learners = learners
        self.n_slices = len(learners) # This might need adjustment based on your setup
        self.n_prbs = n_prbs
        self.exploration_factor = exploration_factor
        self.resource_cost_factor = resource_cost_factor
        self.action = np.array([10] * self.n_slices, dtype=np.int16) # Start with a reasonable action

    def select_action(self, state):
        # This function needs to be adapted based on how many agents you have per slice.
        # Here's a simplified example for ONE slice with two agents: one for safety, one for performance.
        l1_state = state # Assuming state is for one slice for simplicity
        
        safety_agent = self.learners['delay']
        perf_agent = self.learners['throughput']
        
        safe_actions = []
        for a in range(self.n_prbs + 1):
            x = np.append(l1_state, a / self.n_prbs)
            pessimistic_prediction, _ = safety_agent.predict_with_uncertainty(x)
            # Assuming 'delay' is upper-bounded
            if pessimistic_prediction <= safety_agent.sla_threshold:
                safe_actions.append(a)

        best_action_for_slice = 0
        best_final_score = -np.inf
        if safe_actions:
            for a in safe_actions:
                x = np.append(l1_state, a / self.n_prbs)
                prediction, uncertainty = perf_agent.predict_with_uncertainty(x)
                optimistic_score = prediction + self.exploration_factor * uncertainty
                resource_penalty = self.resource_cost_factor * (a / self.n_prbs)
                final_score = optimistic_score - resource_penalty
                if final_score > best_final_score:
                    best_final_score = final_score
                    best_action_for_slice = a
        
        self.action = np.array([best_action_for_slice] * self.n_slices) # Apply same action to all slices for this example
        return self.action, 0

    def update_and_train(self, state, action, info, batch_size=64):
        # Example for one slice
        x_vector = np.append(state, action[0] / self.n_prbs)
        true_delay = info['delay_kpi'] # Assumes info dict has these keys
        true_throughput = info['throughput_kpi']

        self.learners['delay'].replay_buffer.add((x_vector, true_delay))
        self.learners['throughput'].replay_buffer.add((x_vector, true_throughput))

        self.learners['delay'].train_step(batch_size)
        self.learners['throughput'].train_step(batch_size)

    def run(self, system, steps, warmup_steps=200):
        # The run loop is now simpler at a high level
        state, info = system.reset()
        for i in range(steps):
            if i < warmup_steps:
                action = system.action_space.sample()
            else:
                action, _ = self.select_action(state)
            
            next_state, reward, _, _, info = system.step(action)
            self.update_and_train(state, action, info)
            state = next_state
            print(f"Step: {i}, Action: {action}, Reward: {reward:.2f}, Violations: {info.get('total_violations', 0)}")
