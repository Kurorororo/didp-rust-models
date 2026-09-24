use clap::Parser;
use dypdl::prelude::*;
use dypdl_heuristic_search::{
    BeamSearchParameters, CabsParameters, FEvaluatorType, Parameters, create_caasdy,
    create_dual_bound_cabs, create_dual_bound_cahdbs2,
};
use golomb_ruler::{Args, KNOWN_OPTIMAL_COSTS, SolverChoice};
use rpid::timer::Timer;
use std::iter;
use std::{rc::Rc, sync::Arc};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let n = args.n;

    let mut model = Model::default();

    let mark = model.add_object_type("mark", n * n + 2).unwrap();
    let mark_set = model.create_set(mark, &[0]).unwrap();
    let mark_set = model.add_set_variable("mark_set", mark, mark_set).unwrap();
    let distance_set = model.create_set(mark, &[]).unwrap();
    let distance_set = model
        .add_set_variable("distance_set", mark, distance_set)
        .unwrap();
    let last_mark = model.add_element_variable("last_mark", mark, 0).unwrap();
    let n_marks = model.add_element_variable("n_marks", mark, 1).unwrap();

    let element_to_integer = model
        .add_table_1d(
            "element_to_integer",
            (0..=n * n + 1).map(|i| i as i32).collect(),
        )
        .unwrap();

    let mut lower_bounds = Vec::with_capacity(n + 1);
    lower_bounds.extend_from_slice(&KNOWN_OPTIMAL_COSTS);

    while lower_bounds.len() <= n {
        let last = lower_bounds[lower_bounds.len() - 1];
        lower_bounds.push(last + 1);
    }

    let lower_bounds = model.add_table_1d("lower_bounds", lower_bounds).unwrap();

    for i in 1..=(n * n + 1) {
        let mut add_mark = Transition::new(format!("{i}"));
        let last_distance = i - last_mark;
        add_mark
            .set_cost(element_to_integer.element(last_distance.clone()) + IntegerExpression::Cost);

        add_mark.add_effect(mark_set, mark_set.add(i)).unwrap();

        let mut new_distance_set = SetExpression::from(distance_set);

        for j in 0..i {
            new_distance_set = SetElementOperation::<ElementExpression>::add(
                new_distance_set,
                mark_set
                    .contains(j)
                    .if_then_else(i - j, last_distance.clone()),
            );
        }

        add_mark.add_effect(distance_set, new_distance_set).unwrap();
        add_mark.add_effect(last_mark, i).unwrap();
        add_mark.add_effect(n_marks, n_marks + 1).unwrap();

        add_mark.add_precondition(Condition::comparison_e(
            ComparisonOperator::Gt,
            i,
            last_mark,
        ));

        add_mark.add_precondition(
            (Condition::comparison_e(ComparisonOperator::Lt, n_marks, n / 2)
                & Condition::comparison_e(
                    ComparisonOperator::Le,
                    i,
                    (n * n).div_ceil(2) - lower_bounds.element(n / 2 - n_marks),
                ))
                | Condition::comparison_e(ComparisonOperator::Ge, n_marks, n / 2)
                    & Condition::comparison_e(
                        ComparisonOperator::Le,
                        i,
                        n * n + 1 - lower_bounds.element(n - n_marks),
                    ),
        );

        for j in 0..i {
            add_mark.add_precondition(!mark_set.contains(j) | !distance_set.contains(i - j));
        }

        model.add_forward_transition(add_mark).unwrap();
    }

    model
        .add_base_case(vec![Condition::comparison_e(
            ComparisonOperator::Eq,
            n_marks,
            n,
        )])
        .unwrap();

    model
        .add_dual_bound(element_to_integer.element(lower_bounds.element(n - n_marks)))
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
        let marks_str = iter::once(String::from("0"))
            .chain(solution.transitions.iter().map(|t| t.get_full_name()))
            .collect::<Vec<_>>();
        let marks = marks_str
            .iter()
            .map(|i| i.parse().unwrap())
            .collect::<Vec<_>>();
        println!("Marks: {}", marks_str.join(" "));

        if golomb_ruler::validate(n, &marks, cost as usize) {
            println!("The solution is valid");
        } else {
            println!("The solution is invalid");
        }
    }
}
