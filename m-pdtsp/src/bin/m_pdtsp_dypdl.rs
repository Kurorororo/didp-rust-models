use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, create_caasdy,
    create_dual_bound_cabs, create_dual_bound_cahdbs2,
};
use m_pdtsp::{Args, RoundedInstance, SolverChoice};
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

    let instance = Instance::load(&filepath).unwrap();
    let instance = RoundedInstance::try_from(instance).unwrap();

    let mut model = Model::default();

    let n = instance.nodes.len();
    let customer = model.add_object_type("customer", n).unwrap();
    let goal = n - 1;

    let unvisited = (1..goal).collect::<Vec<_>>();
    let unvisited = model.create_set(customer, &unvisited).unwrap();
    let unvisited = model
        .add_set_variable("unvisited", customer, unvisited)
        .unwrap();
    let current = model.add_element_variable("current", customer, 0).unwrap();
    let load = model
        .add_integer_resource_variable("load", true, 0)
        .unwrap();

    let demands = instance
        .demands
        .iter()
        .map(|d| d.iter().sum())
        .collect::<Vec<i32>>();
    let (predecessors, distances) = instance.extract_predecessors_and_filtered_distances();
    let predecessors = predecessors
        .iter()
        .map(|p| {
            let set = p.ones().collect::<Vec<_>>();

            model.create_set(customer, &set).unwrap()
        })
        .collect::<Vec<_>>();
    let connected = distances
        .iter()
        .map(|row| row.iter().map(|&x| x.is_some()).collect())
        .collect();
    let connected = model.add_table_2d("connected", connected).unwrap();
    // Connect impossible suffixes with a finite penalty above every feasible tour.
    let infinity = n as i32
        * distances
            .iter()
            .flatten()
            .filter_map(|&d| d)
            .max()
            .unwrap_or(0)
        + 1;
    let min_goal = distances
        .iter()
        .filter_map(|row| row[goal])
        .min()
        .unwrap_or(infinity);
    let mst_distances = distances
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .map(|(j, &d)| if i == j { 0 } else { d.unwrap_or(infinity) })
                .collect()
        })
        .collect();
    let mst_distances = model.add_table_2d("mst distances", mst_distances).unwrap();
    let distances = distances
        .iter()
        .map(|d| d.iter().map(|&x| x.unwrap_or(0)).collect())
        .collect();
    let distances = model.add_table_2d("distances", distances).unwrap();

    for (next, &d) in demands.iter().enumerate() {
        let mut visit = Transition::new(format!("{next}"));
        visit.set_cost(distances.element(current, next) + IntegerExpression::Cost);

        visit.add_effect(unvisited, unvisited.remove(next)).unwrap();
        visit.add_effect(current, next).unwrap();
        let new_load = load + d;
        visit.add_effect(load, new_load.clone()).unwrap();

        visit.add_precondition(connected.element(current, next));
        visit.add_precondition(unvisited.contains(next));
        visit.add_precondition(Condition::comparison_i(
            ComparisonOperator::Le,
            new_load,
            instance.capacity,
        ));
        visit.add_precondition((unvisited & predecessors[next].clone()).is_empty());

        model.add_forward_transition(visit).unwrap();
    }

    model
        .add_base_case_with_cost(
            vec![connected.element(current, goal), unvisited.is_empty()],
            distances.element(current, goal),
        )
        .unwrap();

    model
        .add_dual_bound(mst_distances.minimum_spanning_tree(unvisited.add(current)) + min_goal)
        .unwrap();

    let parameters = Parameters::<i32> {
        primal_bound: Some(infinity),
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
        let tour = solution
            .transitions
            .iter()
            .map(|t| t.get_full_name())
            .collect::<Vec<_>>();
        println!("Tour: {}", tour.join(" "));
        let tour = tour
            .into_iter()
            .map(|t| t.parse().unwrap())
            .collect::<Vec<_>>();

        if instance.validate(&tour, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
