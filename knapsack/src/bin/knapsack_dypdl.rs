use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    create_caasdy, create_dual_bound_cabs, BeamSearchParameters, CabsParameters, FEvaluatorType,
    Parameters,
};
use knapsack::{Args, Instance, SolverChoice};
use rpid::timer::Timer;
use std::rc::Rc;

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
    let item = model.add_object_type("item", n).unwrap();

    let current = model.add_element_variable("current", item, 0).unwrap();
    let remaining = model
        .add_integer_resource_variable("remaining", false, instance.capacity)
        .unwrap();

    let profits = model
        .add_table_1d("profits", instance.profits.clone())
        .unwrap();
    let weights = model
        .add_table_1d("weights", instance.weights.clone())
        .unwrap();

    let mut pack = Transition::new("pack");
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

    let mut total_profit_after = instance
        .profits
        .iter()
        .rev()
        .scan(0, |acc, &x| {
            *acc += x;

            Some(*acc)
        })
        .collect::<Vec<_>>();
    total_profit_after.reverse();
    total_profit_after.push(0);

    let mut ms = instance
        .profits
        .iter()
        .zip(instance.weights.iter())
        .map(|(&p, &w)| p as f64 / w as f64 + args.epsilon)
        .rev()
        .scan(0.0, |acc, x| {
            if *acc < x {
                *acc = x;
            }

            Some(*acc)
        })
        .collect::<Vec<_>>();
    ms.reverse();
    ms.push(0.0);
    let ms = model.add_table_1d("ms", ms).unwrap();
    model
        .add_dual_bound(IntegerExpression::floor(remaining * ms.element(current)))
        .unwrap();

    let total_profit_after = model
        .add_table_1d("total_profit_after", total_profit_after)
        .unwrap();
    model
        .add_dual_bound(total_profit_after.element(current))
        .unwrap();

    let model = Rc::new(model);

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
            println!("Preparing time: {}s", timer.get_elapsed_time());

            create_dual_bound_cabs(model, parameters, FEvaluatorType::Plus)
        }
        SolverChoice::Astar => {
            println!("Preparing time: {}s", timer.get_elapsed_time());

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
