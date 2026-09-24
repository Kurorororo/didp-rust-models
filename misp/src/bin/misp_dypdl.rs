use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, create_caasdy,
    create_dual_bound_cabs, create_dual_bound_cahdbs2,
};
use misp::{Args, Instance, SolverChoice};
use rpid::timer::Timer;
use std::{rc::Rc, sync::Arc};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();

    let mut model = Model::default();
    model.set_maximize();

    let node = model.add_object_type("node", instance.n).unwrap();

    let candidates = model
        .create_set(node, &(0..instance.n).collect::<Vec<_>>())
        .unwrap();
    let candidates = model
        .add_set_variable("candidates", node, candidates)
        .unwrap();
    let current = model.add_element_variable("current", node, 0).unwrap();

    let neighbors = instance
        .adjacency_list
        .iter()
        .map(|neighbors| model.create_set(node, neighbors).unwrap())
        .collect::<Vec<_>>();
    let neighbors = model.add_table_1d("neighbors", neighbors).unwrap();

    let mut include = Transition::new("include");
    include.set_cost(1 + IntegerExpression::Cost);
    include
        .add_effect(
            candidates,
            candidates.remove(current) - neighbors.element(current),
        )
        .unwrap();
    include.add_effect(current, current + 1).unwrap();
    include.add_precondition(candidates.contains(current));

    model.add_forward_transition(include).unwrap();

    let mut exclude = Transition::new("exclude");
    exclude.set_cost(IntegerExpression::Cost);
    exclude
        .add_effect(candidates, candidates.remove(current))
        .unwrap();
    exclude.add_effect(current, current + 1).unwrap();

    model.add_forward_transition(exclude).unwrap();

    model
        .add_base_case(vec![Condition::comparison_e(
            ComparisonOperator::Eq,
            current,
            instance.n,
        )])
        .unwrap();

    model.add_dual_bound(candidates.len()).unwrap();

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
        let independent_set = solution
            .transitions
            .iter()
            .enumerate()
            .filter_map(|(i, x)| {
                if x.get_full_name() == "include" {
                    Some(i)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        println!("Independent set: {independent_set:?}");

        if instance.validate(&independent_set) {
            if independent_set.len() != cost as usize {
                println!("Cost {cost} != {len}", len = independent_set.len());
                println!("The solution is invalid.");
            } else {
                println!("The solution is valid.");
            }
        } else {
            println!("The solution is invalid.");
        }
    }
}
