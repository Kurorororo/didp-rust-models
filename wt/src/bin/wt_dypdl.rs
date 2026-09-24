use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, create_caasdy,
    create_dual_bound_cabs, create_dual_bound_cahdbs2,
};
use rpid::timer::Timer;
use std::{rc::Rc, sync::Arc};
use wt::{Args, Instance, SolverChoice};

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

    let n = instance.processing_times.len();
    let job = model.add_object_type("job", n).unwrap();

    let scheduled = vec![];
    let scheduled = model.create_set(job, &scheduled).unwrap();
    let scheduled = model.add_set_variable("scheduled", job, scheduled).unwrap();

    let processing_times = model
        .add_table_1d("processing_times", instance.processing_times.clone())
        .unwrap();
    let (predecessors, _) = instance.extract_precedence();
    let predecessors = predecessors
        .iter()
        .map(|p| {
            let set = p.ones().collect::<Vec<_>>();

            model.create_set(job, &set).unwrap()
        })
        .collect::<Vec<_>>();

    let current_time = model
        .add_integer_state_function("current_time", processing_times.sum(scheduled))
        .unwrap();

    for (i, &pi) in instance.processing_times.iter().enumerate() {
        let mut schedule = Transition::new(format!("{i}"));
        let tardiness =
            IntegerExpression::max(current_time.clone() + pi - instance.deadlines[i], 0);
        schedule.set_cost(instance.weights[i] * tardiness + IntegerExpression::Cost);
        schedule.add_effect(scheduled, scheduled.add(i)).unwrap();
        schedule.add_precondition(!scheduled.contains(i));
        schedule
            .add_precondition(SetExpression::from(predecessors[i].clone()).is_subset(scheduled));

        model.add_forward_transition(schedule).unwrap();
    }

    model
        .add_base_case(vec![Condition::comparison_i(
            ComparisonOperator::Eq,
            scheduled.len(),
            n as i32,
        )])
        .unwrap();

    model.add_dual_bound(IntegerExpression::from(0)).unwrap();

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
            .map(|t| t.get_full_name())
            .collect::<Vec<_>>();
        println!("Schedule: {schedule}", schedule = sequence.join(" "));
        let sequence = sequence
            .iter()
            .map(|s| s.parse().unwrap())
            .collect::<Vec<_>>();

        if instance.validate(&sequence, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
