use clap::Parser;
use optw::rpid_model::Optw;
use optw::{Args, Instance, RoundedInstance, SolverChoice};
use rpid::prelude::*;
use rpid::{io, solvers, timer::Timer};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();
    let rounded_instance = RoundedInstance::new(instance, args.round_to);
    let optw = Optw::new(rounded_instance.clone(), args.epsilon);

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };

    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = if args.threads.get() > 1 {
                solvers::create_parallel_cabs(optw, parameters, cabs_parameters, args.threads.get())
            } else {
                solvers::create_cabs(optw, parameters, cabs_parameters)
            };
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(optw, parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(profit) = solution.cost {
        let tour = solution
            .transitions
            .into_iter()
            .filter(|&i| i < rounded_instance.vertices.len())
            .collect::<Vec<_>>();
        rounded_instance.print_solution(&tour);

        if rounded_instance.validate(&tour, profit) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
