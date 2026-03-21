def _compute_outcome(action: str, event: MicroEvent, event_idx: int) -> float:
    """Compute synthetic outcome for an action.
    
    Returns a score from 0.0 to 1.0 where:
    - 1.0: Perfect match between action and event type
    - 0.3: Mismatch between action and event type  
    - 0.0: Abstention
    
    Logic:
    - Page events (types 2,3): page actions get 1.0, others 0.3
    - Batch events (14,15): batch actions get 1.0, others 0.3
    - Even PIDs: numa actions get 1.0, others 0.3
    - Odd PIDs: boundary actions get 1.0, others 0.3
    """
    if action == "abstain":
        return 0.0
    
    # Map expert names to head names
    expert_to_head = {
        'page_transition': 'page',
        'cow_fork': 'page', 
        'reclaim_hotness': 'page',
        'locality_pattern': 'batch',
        'fault_burst': 'page',
        'boundary_control': 'boundary',
        'kv_policy': 'kv',
        'numa_placement': 'numa',
    }
    
    # Get the head for this action
    head = expert_to_head.get(action, action)  # action could be expert name or head name
    
    # Page events (faults) favor page actions
    if event.event_type in [2, 3]:  # PAGE_FAULT, COW_FAULT
        return 1.0 if head == "page" else 0.3
    
    # Batch events favor batch actions  
    if event.event_type in [14, 15]:  # QUEUE, KV_POLICY
        return 1.0 if head == "batch" else 0.3
    
    # For other events, use PID-based logic
    if event.pid % 2 == 0:
        # Even PIDs favor numa actions
        return 1.0 if head == "numa" else 0.3
    else:
        # Odd PIDs favor boundary actions
        return 1.0 if head == "boundary" else 0.3