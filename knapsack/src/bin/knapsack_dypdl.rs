use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, create_caasdy,
    create_dual_bound_cabs, create_dual_bound_cahdbs2,
};
use knapsack::{Args, Instance, SolverChoice};
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

    let n = instance.profits.len();
    let item = model.add_object_type("item", n + 1).unwrap();

    let current = model.add_element_variable("current", item, 0).unwrap();
    let remaining = model
        .add_integer_resource_variable("remaining", false, instance.capacity)
        .unwrap();

    let profits = model
        .add_table_1d(
            "profits",
            instance.profits.iter().copied().chain([0]).collect(),
        )
        .unwrap();
    let weights = model
        .add_table_1d(
            "weights",
            instance.weights.iter().copied().chain([0]).collect(),
        )
        .unwrap();

    let mut pack = Transition::new("pack");
    pack.add_precondition(Condition::comparison_e(ComparisonOperator::Lt, current, n));
    pack.set_cost(profits.element(current) + IntegerExpression::Cost);
    pack.add_effect(current, current + 1).unwrap();

    pack.add_effect(remaining, remaining - weights.element(current))
        .unwrap();
    pack.add_precondition(Condition::comparison_i(
        ComparisonOperator::Ge,
        remaining,
        weights.element(current),
    ));

    model.add_forward_transition(pack).unwrap();

    let mut ignore = Transition::new("ignore");
    ignore.add_precondition(Condition::comparison_e(ComparisonOperator::Lt, current, n));
    ignore.set_cost(IntegerExpression::Cost);
    ignore.add_effect(current, current + 1).unwrap();
    model.add_forward_transition(ignore).unwrap();

    model
        .add_base_case(vec![Condition::comparison_e(
            ComparisonOperator::Eq,
            current,
            n,
        )])
        .unwrap();

    let rewards = model
        .add_table_1d(
            "rewards",
            instance
                .profits
                .iter()
                .map(|&p| std::cmp::max(p, 0))
                .chain([0])
                .collect(),
        )
        .unwrap();
    let suffix = (0..=n)
        .map(|i| model.create_set(item, &(i..n).collect::<Vec<_>>()).unwrap())
        .collect();
    let suffix = model.add_table_1d("suffix", suffix).unwrap();
    let bound = ContinuousExpression::fractional_knapsack_with_integer_tables(
        suffix.element(current),
        remaining,
        rewards,
        weights,
    );
    model
        .add_dual_bound(IntegerExpression::floor(bound + args.epsilon))
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

    if let Some(profit) = solution.cost {
        let packed_items = solution
            .transitions
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                if t.get_full_name() == "pack" {
                    Some(i)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        instance.print_solution(&packed_items);

        if instance.validate(&packed_items, profit) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
