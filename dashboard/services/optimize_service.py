from dashboard.data.mock_data import allocate_budget, budget_allocation_by_segment


def get_budget_result(customers, budget: int):
    selected, summary = allocate_budget(customers, budget)
    segment_allocation = budget_allocation_by_segment(selected)
    return selected, summary, segment_allocation