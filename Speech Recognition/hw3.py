import math

def read_state_weights_txt(filename):
    with open(filename, 'r') as f:
        lines = f.read().strip().split('\n')
 
    states_string = lines[1].split()[0] 
    num_states = int(states_string) 
    state_dictionary = {}
    total_weight = 0

    for line in lines[2:]:
    
        state, weight = line.strip().split()
        state = state.strip('"')
        weight = int(weight)
        state_dictionary[state] = weight
        total_weight += weight
    
    for state in state_dictionary:
        prev_weight = state_dictionary[state]
        normalized_weight = prev_weight/total_weight
        state_dictionary[state] = normalized_weight
    return state_dictionary

def read_state_observation_weights(filename):
 
    state_obs_weights = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_lines = lines[2:]
    num_pairs, num_states, num_observations, default_weight = lines[1].split()
    default_weight = int(default_weight)


    for line in data_lines:

        info_line = line.strip().split()

        if len(info_line) == 2:
            state, obs = info_line
            weight = default_weight
        else:
            state, obs, weight = info_line
            weight = int(weight)

        #state, obs, weight = line.strip().split()
        #weight = int(weight)         

        state = state.strip('"')     
        obs = obs.strip('"')         

        if state not in state_obs_weights:
            state_obs_weights[state] = {}
        state_obs_weights[state][obs] = weight

    emission_probs = {}
    for state, obs_dict in state_obs_weights.items():
        total_weight = sum(obs_dict.values())
        emission_probs[state] = {}
        for obs, weight in obs_dict.items():
            emission_probs[state][obs] = weight / total_weight

    return emission_probs


def read_state_action_state_weights(filename):
    trans_weights = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    data_lines = lines[2:]
    
    num_triples, num_states, num_actions, default_weight = lines[1].split()
    num_triples = int(num_triples)
    num_states = int(num_states)
    num_actions = int(num_actions)
    default_weight = int(default_weight)

    for line in data_lines:

        info_line = line.strip().split()
        if len(info_line) == 3:
             initial_state, action, next_state = info_line
             weight = default_weight
        else:
            initial_state, action, next_state, weight = info_line
            weight = int(weight)

        initial_state = initial_state.strip('"')  
        action = action.strip('"')                
        next_state = next_state.strip('"')       
        weight = int(weight)                     
        
        key = (initial_state, action)
        
        if key not in trans_weights:
            trans_weights[key] = {}
        trans_weights[key][next_state] = weight
    
    transition_probs = {}
    for key, next_states in trans_weights.items():
        total_weight = sum(next_states.values())
        transition_probs[key] = {}
        for next_state, weight in next_states.items():
            transition_probs[key][next_state] = weight / total_weight
   
    return transition_probs


def read_observation_actions(filename):
    observations = []
    actions = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_lines = lines[2:]
    

    for line in data_lines:
        act_obs = line.strip().split()
        if act_obs:

            if len(act_obs) == 2:
                obs = act_obs[0].strip('"')
                action = act_obs[1].strip('"')
                observations.append(obs)
                actions.append(action)
            elif len(act_obs) == 1:
                obs = act_obs[0].strip('"')
                observations.append(obs)
                actions.append(None)
    
    return observations, actions



def viterbi_algorithm(observations, actions, state_weights, trans_weights, state_obs_weights, states):
  
    num_obs = len(observations)

    Viterbi = [{} for _ in range(num_obs)]
    paths = {}

    for state in states:
        if observations[0] in state_obs_weights[state]:
            emission_prob = state_obs_weights[state][observations[0]]
        else:
            emission_prob = 0

        Viterbi[0][state] = state_weights[state] * emission_prob
        paths[state] = [state]
        #print(paths[state],' ',state)

    for t in range(1, num_obs):
        current_paths = {}
        obs = observations[t]
        action = None
        if t - 1 < len(actions):
            action = actions[t - 1]
            #print(action)
        for curr_state in states:
            max_prob = -1
            best_prev_state = None

            for prev_state in states:
                if action is None:
                    
                    if prev_state == curr_state:
                        trans_prob = 1.0
                    else:
                        trans_prob = 0.0
                else:
                    trans_key = (prev_state, action)
                    if trans_key in trans_weights and curr_state in trans_weights[trans_key]:
                        trans_prob = trans_weights[trans_key][curr_state]
                    else:
                        trans_prob = 0.0

                if obs in state_obs_weights[curr_state]:
                    emision_p = state_obs_weights[curr_state][obs]
                else:
                    emision_p = 0

                prob = Viterbi[t - 1][prev_state] * trans_prob * emision_p

                if prob > max_prob:
                    max_prob = prob
                    best_prev_state = prev_state

            Viterbi[t][curr_state] = max_prob
            current_paths[curr_state] = paths[best_prev_state] + [curr_state]
            #print(f"t={t}, curr_state={curr_state}, prob={max_prob:.6f}, best_prev_state={best_prev_state}")

        paths = current_paths

    return Viterbi, paths

def Viterbi_get_best_path(Viterbi, states, paths):
    max_prob = -1
    best_final_state = None
    vit_len = len(Viterbi)-1
    all_paths = paths
    for state in states:
        if Viterbi[vit_len][state] > max_prob:
            max_prob = Viterbi[vit_len][state]
            best_final_state = state
    best_path = all_paths[best_final_state]
    return best_path


if __name__ == '__main__':
    state_weights = read_state_weights_txt('state_weights.txt')
    state_obs_weights = read_state_observation_weights('state_observation_weights.txt')
    trans_weights = read_state_action_state_weights('state_action_state_weights.txt')
    observations, acts_extracted = read_observation_actions('observation_actions.txt')
    states = list(state_weights.keys())

    actions = []
    for act in acts_extracted:
        if act is not None:
            actions.append(act)


    Viterbi, paths = viterbi_algorithm(observations, actions, state_weights, trans_weights, state_obs_weights,states)
    result_path = Viterbi_get_best_path(Viterbi,states,paths)
 
    with open('states.txt', 'w') as f:
        f.write('states\n')
        f.write(f'{len(result_path)}\n')
        for state in result_path:
            f.write(f'"{state}"\n')