use crate::RoundedInstance;
use dypdl::prelude::*;
use rpid::algorithms;

pub fn create_model(instance: &RoundedInstance, epsilon: f64) -> Model {
    assert!(
        epsilon.is_finite() && epsilon >= 0.0,
        "epsilon must be finite and nonnegative"
    );
    let mut model = Model::default();
    model.set_maximize();
    let n = instance.vertices.len();
    let customer = model.add_object_type("customer", n + 1).unwrap();
    let shortest = crate::compute_pairwise_shortest_path_costs(&instance.distances);
    let initial_time = std::cmp::max(instance.opening[0], 0);
    let initially_reachable = (1..n)
        .filter(|&i| {
            let earliest = std::cmp::max(initial_time + shortest[0][i], instance.opening[i]);
            earliest <= instance.closing[i] && earliest + shortest[i][0] <= instance.closing[0]
        })
        .collect::<Vec<_>>();
    let reachable = model.create_set(customer, &initially_reachable).unwrap();
    let reachable = model
        .add_set_resource_variable("reachable", customer, false, reachable)
        .unwrap();
    let current = model.add_element_variable("current", customer, 0).unwrap();
    let time = model
        .add_integer_resource_variable("time", true, initial_time)
        .unwrap();
    let mut travel = instance.distances.clone();
    travel.push(vec![0; n]);
    let travel = model.add_table_2d("travel", travel).unwrap();
    let mut shortest_rows = shortest.clone();
    shortest_rows.push(vec![0; n]);
    let shortest_table = model.add_table_2d("shortest", shortest_rows).unwrap();
    let opening = model
        .add_table_1d("opening", instance.opening.clone())
        .unwrap();
    let closing = model
        .add_table_1d("closing", instance.closing.clone())
        .unwrap();
    let shortest_return = model
        .add_table_1d(
            "shortest return",
            shortest.iter().map(|row| row[0]).collect(),
        )
        .unwrap();
    model
        .add_state_constraint(Condition::comparison_i(
            ComparisonOperator::Le,
            time,
            instance.closing[0],
        ))
        .unwrap();
    let finished = Condition::comparison_e(ComparisonOperator::Eq, current, n);
    model.add_base_case(vec![finished.clone()]).unwrap();
    let x = model.add_local_variable("x").unwrap();
    for (next, shortest_row) in shortest.iter().enumerate().skip(1) {
        let time_next = model
            .add_integer_state_function(
                format!("time after {next}"),
                IntegerExpression::max(
                    time + travel.element(current, next),
                    instance.opening[next],
                ),
            )
            .unwrap();
        let earliest = IntegerExpression::max(
            time_next.clone() + shortest_table.element(next, x),
            opening.element(x),
        );
        let can_reach =
            Condition::comparison_i(ComparisonOperator::Le, earliest.clone(), closing.element(x))
                & Condition::comparison_i(
                    ComparisonOperator::Le,
                    earliest + shortest_return.element(x),
                    instance.closing[0],
                );
        let mut visit = Transition::new(format!("{next}"));
        visit.set_cost(instance.profits[next] + IntegerExpression::Cost);
        visit.add_effect(current, next).unwrap();
        visit.add_effect(time, time_next.clone()).unwrap();
        visit
            .add_effect(reachable, reachable.remove(next).filter(x, can_reach))
            .unwrap();
        visit.add_precondition(!finished.clone());
        visit.add_precondition(reachable.contains(next));
        visit.add_precondition(Condition::comparison_i(
            ComparisonOperator::Le,
            time_next.clone(),
            instance.closing[next],
        ));
        visit.add_precondition(Condition::comparison_i(
            ComparisonOperator::Le,
            time_next + shortest_row[0],
            instance.closing[0],
        ));
        model.add_forward_transition(visit).unwrap();
    }
    let mut finish = Transition::new(format!("{n}"));
    finish.set_cost(IntegerExpression::Cost);
    finish.add_effect(current, n).unwrap();
    finish
        .add_effect(time, time + travel.element(current, 0))
        .unwrap();
    finish
        .add_effect(reachable, model.create_set(customer, &[]).unwrap())
        .unwrap();
    finish.add_precondition(!finished.clone());
    finish.add_precondition(Condition::comparison_i(
        ComparisonOperator::Le,
        time + travel.element(current, 0),
        instance.closing[0],
    ));
    model.add_forward_transition(finish).unwrap();

    let rewards = model
        .add_table_1d(
            "rewards",
            instance
                .profits
                .iter()
                .map(|&p| std::cmp::max(p, 0))
                .collect(),
        )
        .unwrap();
    let min_from = algorithms::take_row_wise_min_without_diagonal(&instance.distances)
        .map(|d| d.unwrap_or(0))
        .collect::<Vec<_>>();
    let min_to = algorithms::take_column_wise_min_without_diagonal(&instance.distances)
        .map(|d| d.unwrap_or(0))
        .collect::<Vec<_>>();
    let from_table = model
        .add_table_1d("min from", min_from.iter().copied().chain([0]).collect())
        .unwrap();
    for (name, weights, capacity) in [
        (
            "weights from",
            min_from,
            instance.closing[0] - time - from_table.element(current),
        ),
        (
            "weights to",
            min_to.clone(),
            instance.closing[0] - time - min_to[0],
        ),
    ] {
        let weights = model.add_table_1d(name, weights).unwrap();
        let bound = ContinuousExpression::fractional_knapsack_with_integer_tables(
            reachable,
            capacity.max(0),
            rewards,
            weights,
        );
        model
            .add_dual_bound(IfThenElse::<IntegerExpression>::if_then_else(
                finished.clone(),
                0,
                IntegerExpression::floor(bound + epsilon),
            ))
            .unwrap();
    }
    model
}
