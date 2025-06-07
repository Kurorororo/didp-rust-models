use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    create_caasdy, create_dual_bound_cabs, BeamSearchParameters, CabsParameters, FEvaluatorType,
    Parameters,
};
use rpid::{algorithms, timer::Timer};
use std::rc::Rc;
use tsptw::{Args, Instance, SimplificationChoice, SolverChoice};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let mut instance = Instance::read_from_file(&args.input_file).unwrap();

    match args.simplification_level {
        SimplificationChoice::None => {}
        SimplificationChoice::Cheap => {
            instance.simplify(false);
        }
        SimplificationChoice::Expensive => {
            instance.simplify(true);
        }
    }

    let mut model = Model::default();

    let n = instance.a.len();
    let customer = model.add_object_type("customer", n).unwrap();

    let unvisited = (1..n).collect::<Vec<_>>();
    let unvisited = model.create_set(customer, &unvisited).unwrap();
    let unvisited = model
        .add_set_variable("unvisited", customer, unvisited)
        .unwrap();
    let current = model.add_element_variable("current", customer, 0).unwrap();
    let time = model
        .add_integer_resource_variable("time", true, 0)
        .unwrap();

    let c = instance
        .c
        .iter()
        .map(|row| row.iter().map(|&x| x.unwrap_or(0)).collect())
        .collect();
    let c = model.add_table_2d("c", c).unwrap();
    let connected = instance
        .c
        .iter()
        .map(|row| row.iter().map(|&x| x.is_some()).collect())
        .collect();
    let connected = model.add_table_2d("connected", connected).unwrap();

    for next in 1..n {
        let mut visit = Transition::new(format!("{}", next));

        let arrival_time = time + c.element(current, next);
        let start_time = IntegerExpression::max(arrival_time.clone(), instance.a[next]);

        if args.minimize_makespan {
            visit.set_cost(
                IntegerExpression::max(c.element(current, next), instance.a[next] - time)
                    + IntegerExpression::Cost,
            );
        } else {
            visit.set_cost(c.element(current, next) + IntegerExpression::Cost);
        }

        visit.add_effect(unvisited, unvisited.remove(next)).unwrap();
        visit.add_effect(current, next).unwrap();
        visit.add_effect(time, start_time).unwrap();

        if args.simplification_level == SimplificationChoice::Expensive {
            visit.add_precondition(connected.element(current, next));
        }

        visit.add_precondition(unvisited.contains(next));
        visit.add_precondition(Condition::comparison_i(
            ComparisonOperator::Le,
            arrival_time,
            instance.b[next],
        ));

        model.add_forward_transition(visit).unwrap();
    }

    model
        .add_base_case_with_cost(vec![unvisited.is_empty()], c.element(current, 0))
        .unwrap();

    let mut c = instance.c.clone();
    c.iter_mut().for_each(|row| {
        row[0] = None;
    });
    let c_star = algorithms::compute_pairwise_shortest_path_costs_with_option(&c);
    let c_star = c_star
        .into_iter()
        .map(|row| row.iter().map(|&x| x.unwrap_or(0)).collect())
        .collect();
    let c_star = model.add_table_2d("c_star", c_star).unwrap();

    for next in 1..n {
        let arrival_time = time + c_star.element(current, next);
        let on_time =
            Condition::comparison_i(ComparisonOperator::Le, arrival_time, instance.b[next]);
        model
            .add_state_constraint(!unvisited.contains(next) | on_time)
            .unwrap();
    }

    let min_to = algorithms::take_column_wise_min_with_option(&instance.c)
        .map(|x| x.unwrap())
        .collect::<Vec<_>>();
    let min_to = model.add_table_1d("min_to", min_to).unwrap();
    model
        .add_dual_bound(min_to.sum(unvisited) + min_to.element(0))
        .unwrap();

    let min_from = algorithms::take_row_wise_min_with_option(&instance.c)
        .map(|x| x.unwrap())
        .collect::<Vec<_>>();
    let min_from = model.add_table_1d("min_from", min_from).unwrap();
    model
        .add_dual_bound(min_from.sum(unvisited) + min_from.element(current))
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

        if (args.minimize_makespan && instance.validate_makespan(&tour, cost))
            || instance.validate(&tour, cost)
        {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
