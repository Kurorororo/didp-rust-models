use clap::Parser;
use cvrp::{Args, RoundedInstance, SolverChoice};
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, create_caasdy,
    create_dual_bound_cabs, create_dual_bound_cahdbs2,
};
use regex::Regex;
use rpid::timer::Timer;
use std::{rc::Rc, sync::Arc};
use tsplib_parser::Instance;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let filepath = args.input_file;
    let filename = filepath.split('/').next_back().unwrap();

    let re = Regex::new(r".+k(\d+).+").unwrap();
    let n_vehicles = re.captures(filename).unwrap()[1].parse().unwrap();

    let instance = Instance::load(&filepath).unwrap();
    let mut instance = RoundedInstance::new(instance, n_vehicles).unwrap();
    let n_vehicles = n_vehicles as i32;

    if args.reduce_edges {
        instance.reduce_edges();
    }

    let depot = instance.depot;

    let mut model = Model::default();

    let n = instance.nodes.len();
    let customer = model.add_object_type("customer", n).unwrap();

    let unvisited = (0..n).filter(|&i| i != depot).collect::<Vec<_>>();
    let unvisited = model.create_set(customer, &unvisited).unwrap();
    let unvisited = model
        .add_set_variable("unvisited", customer, unvisited)
        .unwrap();
    let current = model
        .add_element_variable("current", customer, depot)
        .unwrap();
    let load = model
        .add_integer_resource_variable("load", true, 0)
        .unwrap();
    let k = model.add_integer_resource_variable("k", true, 1).unwrap();

    let distances = instance
        .distances
        .iter()
        .map(|row| row.iter().map(|&x| x.unwrap_or(0)).collect())
        .collect();
    let distances = model.add_table_2d("distances", distances).unwrap();

    let connected = model
        .add_table_2d(
            "connected",
            instance
                .distances
                .iter()
                .map(|row| row.iter().map(|d| d.is_some()).collect())
                .collect(),
        )
        .unwrap();

    for next in (0..n).filter(|&i| i != depot) {
        let mut visit = Transition::new(format!("{next}"));
        visit.set_cost(distances.element(current, next) + IntegerExpression::Cost);

        visit.add_effect(unvisited, unvisited.remove(next)).unwrap();
        visit.add_effect(current, next).unwrap();
        visit
            .add_effect(load, load + instance.demands[next])
            .unwrap();

        visit.add_precondition(unvisited.contains(next));
        visit.add_precondition(connected.element(current, next));
        visit.add_precondition(Condition::comparison_i(
            ComparisonOperator::Le,
            load + instance.demands[next],
            instance.capacity,
        ));

        model.add_forward_transition(visit).unwrap();
    }

    let distances_via_depot = instance
        .distances
        .iter()
        .map(|row| {
            {
                (0..n).map(|j| {
                    if let (Some(distance_to_depot), Some(distance_from_depot)) =
                        (row[depot], instance.distances[depot][j])
                    {
                        distance_to_depot + distance_from_depot
                    } else {
                        0
                    }
                })
            }
            .collect()
        })
        .collect();
    let distances_via_depot = model
        .add_table_2d("distances_via_depot", distances_via_depot)
        .unwrap();

    for next in (0..n).filter(|&i| i != depot) {
        let mut visit_via_depot = Transition::new(format!("{}", n + next));
        visit_via_depot
            .set_cost(distances_via_depot.element(current, next) + IntegerExpression::Cost);

        visit_via_depot
            .add_effect(unvisited, unvisited.remove(next))
            .unwrap();
        visit_via_depot.add_effect(current, next).unwrap();
        visit_via_depot
            .add_effect(load, instance.demands[next])
            .unwrap();
        visit_via_depot.add_effect(k, k + 1).unwrap();

        visit_via_depot.add_precondition(unvisited.contains(next));
        visit_via_depot.add_precondition(connected.element(current, depot));
        visit_via_depot.add_precondition(connected.element(depot, next));
        visit_via_depot.add_precondition(Condition::comparison_i(
            ComparisonOperator::Le,
            instance.demands[next],
            instance.capacity,
        ));
        visit_via_depot.add_precondition(Condition::comparison_e(
            ComparisonOperator::Ne,
            current,
            depot,
        ));
        visit_via_depot.add_precondition(Condition::comparison_i(
            ComparisonOperator::Lt,
            k,
            n_vehicles,
        ));

        model.add_forward_transition(visit_via_depot).unwrap();
    }

    model
        .add_base_case_with_cost(
            vec![
                unvisited.is_empty(),
                connected.element(current, depot)
                    | Condition::comparison_e(ComparisonOperator::Eq, current, depot),
            ],
            distances.element(current, depot),
        )
        .unwrap();

    let demands = model
        .add_table_1d("demands", instance.demands.clone())
        .unwrap();
    let total_remaining_capacity = (n_vehicles - k) * instance.capacity + instance.capacity;
    let total_remaining_demand = load + demands.sum(unvisited);
    model
        .add_state_constraint(Condition::comparison_i(
            ComparisonOperator::Ge,
            total_remaining_capacity,
            total_remaining_demand,
        ))
        .unwrap();

    // A compressed edge may go via the depot, even when rounding makes it
    // cheaper than the direct edge. Missing edges receive a finite penalty.
    let infinity = (n as i32 + n_vehicles)
        * instance
            .distances
            .iter()
            .flatten()
            .filter_map(|&d| d)
            .max()
            .unwrap_or(0)
        + 1;
    let mst_distances = instance
        .distances
        .iter()
        .enumerate()
        .map(|(i, row)| {
            (0..n)
                .map(|j| {
                    if i == j {
                        return 0;
                    }
                    let via = row[depot]
                        .zip(instance.distances[depot][j])
                        .map(|(a, b)| a + b);
                    row[j].into_iter().chain(via).min().unwrap_or(infinity)
                })
                .collect()
        })
        .collect();
    let mst_distances = model.add_table_2d("mst distances", mst_distances).unwrap();
    let min_return = instance
        .distances
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != depot)
        .filter_map(|(_, row)| row[depot])
        .min()
        .unwrap_or(0);
    model
        .add_dual_bound(IfThenElse::<IntegerExpression>::if_then_else(
            unvisited.is_empty(),
            distances.element(current, depot),
            mst_distances.minimum_spanning_tree(unvisited.add(current)) + min_return,
        ))
        .unwrap();

    let parameters = Parameters::<i32> {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };

    let mut solver = match args.solver {
        SolverChoice::Cabs => {
            let beam_search_parameters = BeamSearchParameters {
                parameters,
                ..Default::default()
            };
            let parameters = CabsParameters {
                beam_search_parameters,
                ..Default::default()
            };
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());

            if args.threads.get() > 1 {
                let model = Arc::new(model);
                create_dual_bound_cahdbs2(
                    model,
                    parameters,
                    FEvaluatorType::Plus,
                    args.threads.get(),
                )
            } else {
                let model = Rc::new(model);
                create_dual_bound_cabs(model, parameters, FEvaluatorType::Plus)
            }
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());

            let model = Rc::new(model);
            create_caasdy(model, parameters, FEvaluatorType::Plus)
        }
    };

    let solution =
        io_util::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap();
    io_util::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let mut tours = vec![vec![]];

        for transition in solution.transitions {
            let i = transition.get_full_name().parse::<usize>().unwrap();

            if i >= n {
                tours.push(vec![i - n]);
            } else {
                tours.last_mut().unwrap().push(i);
            }
        }

        instance.print_solution(&tours);

        if instance.validate(&tours, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
