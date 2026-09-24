use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, create_caasdy,
    create_dual_bound_cabs, create_dual_bound_cahdbs2,
};
use rpid::timer::Timer;
use salbp_1::{Args, Instance, SolverChoice};
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

    let n = instance.task_times.len();
    let task = model.add_object_type("task", n).unwrap();

    let remaining = model
        .add_integer_resource_variable("remaining", false, 0)
        .unwrap();
    let unscheduled = (0..n).collect::<Vec<_>>();
    let unscheduled = model.create_set(task, &unscheduled).unwrap();
    let unscheduled = model
        .add_set_variable("unscheduled", task, unscheduled)
        .unwrap();

    let predecessors = instance
        .predecessors
        .iter()
        .map(|p| {
            let predecessors = p.ones().collect::<Vec<_>>();
            model.create_set(task, &predecessors).unwrap()
        })
        .collect::<Vec<_>>();

    for (i, &wi) in instance.task_times.iter().enumerate() {
        let mut schedule = Transition::new(format!("{i}"));
        schedule.set_cost(IntegerExpression::Cost);

        schedule.add_effect(remaining, remaining - wi).unwrap();
        schedule
            .add_effect(unscheduled, unscheduled.remove(i))
            .unwrap();

        schedule.add_precondition(unscheduled.contains(i));
        schedule.add_precondition(Condition::comparison_i(
            ComparisonOperator::Le,
            wi,
            remaining,
        ));
        schedule.add_precondition((unscheduled & predecessors[i].clone()).is_empty());

        model.add_forward_transition(schedule).unwrap();
    }

    let mut open = Transition::new(format!("{n}"));
    open.set_cost(1 + IntegerExpression::Cost);
    open.add_effect(remaining, instance.cycle_time).unwrap();

    for (i, &wi) in instance.task_times.iter().enumerate() {
        open.add_precondition(
            !unscheduled.contains(i)
                | Condition::comparison_i(ComparisonOperator::Gt, wi, remaining)
                | !(unscheduled & predecessors[i].clone()).is_empty(),
        );
    }

    model.add_forward_forced_transition(open).unwrap();

    model.add_base_case(vec![unscheduled.is_empty()]).unwrap();

    let weights = model
        .add_table_1d("weights", instance.task_times.clone())
        .unwrap();
    model
        .add_dual_bound(IntegerExpression::ceil(
            ContinuousExpression::from(weights.sum(unscheduled) - remaining)
                / ContinuousExpression::from(instance.cycle_time),
        ))
        .unwrap();

    let lb2_weight1 = instance
        .task_times
        .iter()
        .map(|&x| if 2 * x > instance.cycle_time { 1 } else { 0 })
        .collect();
    let lb2_weight1 = model.add_table_1d("lb2_weight1", lb2_weight1).unwrap();
    let lb2_weight2 = instance
        .task_times
        .iter()
        .map(|&x| {
            if 2 * x == instance.cycle_time {
                0.5
            } else {
                0.0
            }
        })
        .collect();
    let lb2_weight2 = model.add_table_1d("lb2_weight2", lb2_weight2).unwrap();
    let remaining_ge_half =
        Condition::comparison_i(ComparisonOperator::Ge, 2 * remaining, instance.cycle_time);
    model
        .add_dual_bound(
            lb2_weight1.sum(unscheduled) + IntegerExpression::ceil(lb2_weight2.sum(unscheduled))
                - IfThenElse::<IntegerExpression>::if_then_else(remaining_ge_half, 1, 0),
        )
        .unwrap();

    let lb3_weight = instance
        .task_times
        .iter()
        .map(|&x| {
            if 3 * x > 2 * instance.cycle_time {
                1.0
            } else if 3 * x == 2 * instance.cycle_time {
                0.666
            } else if 3 * x > instance.cycle_time {
                0.5
            } else if 3 * x == instance.cycle_time {
                0.333
            } else {
                0.0
            }
        })
        .collect();
    let lb3_weight = model.add_table_1d("lb3_weight", lb3_weight).unwrap();
    let remaining_ge_one_third =
        Condition::comparison_i(ComparisonOperator::Ge, 3 * remaining, instance.cycle_time);
    model
        .add_dual_bound(
            IntegerExpression::ceil(lb3_weight.sum(unscheduled))
                - IfThenElse::<IntegerExpression>::if_then_else(remaining_ge_one_third, 1, 0),
        )
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
        let sequence = solution
            .transitions
            .iter()
            .filter_map(|t| {
                let i = t.get_full_name().parse().unwrap();
                if i < n { Some(i) } else { None }
            })
            .collect::<Vec<_>>();
        instance.print_solution(&sequence);

        if instance.validate(&sequence, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
