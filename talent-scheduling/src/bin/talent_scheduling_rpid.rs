use clap::Parser;
use fixedbitset::FixedBitSet;
use rpid::prelude::*;
use rpid::{io, solvers, timer::Timer};
use talent_scheduling::{Args, Instance, SolverChoice};

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Clone)]
struct TalentScheduling {
    instance: Instance,
    simplified_instance: Instance,
    single_actor_cost: i32,
    scene_to_originals: Vec<Vec<usize>>,
    scene_to_actors: Vec<FixedBitSet>,
    scene_to_base_cost: Vec<i32>,
    scene_to_subsumption_candidates: Vec<Vec<usize>>,
}

impl From<Instance> for TalentScheduling {
    fn from(instance: Instance) -> Self {
        let (simplified_instance, single_actor_cost, scene_to_originals) = instance.simplify();

        let scene_to_actors = simplified_instance.create_scene_to_actors();
        let scene_to_base_cost = scene_to_actors
            .iter()
            .enumerate()
            .map(|(i, actors)| {
                actors
                    .ones()
                    .map(|j| simplified_instance.actor_to_cost[j])
                    .sum::<i32>()
                    * simplified_instance.scene_to_duration[i]
            })
            .collect();

        let n = simplified_instance.scene_to_duration.len();
        let mut scene_to_subsumption_candidates = vec![vec![]; n];

        for i in 0..n {
            for j in 0..n {
                if i != j && scene_to_actors[i].is_subset(&scene_to_actors[j]) {
                    scene_to_subsumption_candidates[i].push(j);
                }
            }
        }

        Self {
            instance,
            simplified_instance,
            single_actor_cost,
            scene_to_originals,
            scene_to_actors,
            scene_to_base_cost,
            scene_to_subsumption_candidates,
        }
    }
}

impl TalentScheduling {
    fn reconstruct_solution(&self, solution: &[usize]) -> Vec<usize> {
        solution
            .iter()
            .flat_map(|&scene| self.scene_to_originals[scene].iter().cloned())
            .collect()
    }
}

impl Dp for TalentScheduling {
    type State = FixedBitSet;
    type CostType = i32;
    type Label = usize;

    fn get_target(&self) -> Self::State {
        let n = self.simplified_instance.scene_to_duration.len();
        let mut remaining = FixedBitSet::with_capacity(n);
        remaining.insert_range(..);

        remaining
    }

    fn get_successors(
        &self,
        remaining: &Self::State,
    ) -> impl IntoIterator<Item = (Self::State, Self::CostType, Self::Label)> {
        let m = self.simplified_instance.actor_to_cost.len();
        let n = self.simplified_instance.scene_to_duration.len();
        let mut standby = FixedBitSet::with_capacity(m);
        let mut came = FixedBitSet::with_capacity(m);

        (0..n).for_each(|i| {
            if remaining.contains(i) {
                standby.union_with(&self.scene_to_actors[i]);
            } else {
                came.union_with(&self.scene_to_actors[i]);
            }
        });

        standby.intersect_with(&came);

        let mut successors = Vec::with_capacity(remaining.count_ones(..));

        for i in remaining.ones() {
            if self.scene_to_actors[i] == standby {
                let mut successor_remaining = remaining.clone();
                successor_remaining.remove(i);

                return vec![(successor_remaining, self.scene_to_base_cost[i], i)];
            }

            let mut came_and_come = came.clone();
            came_and_come.union_with(&self.scene_to_actors[i]);

            if self.scene_to_subsumption_candidates[i].iter().any(|&j| {
                remaining.contains(j) && self.scene_to_actors[j].is_subset(&came_and_come)
            }) {
                continue;
            }

            let mut successor_remaining = remaining.clone();
            successor_remaining.remove(i);

            let weight = self.simplified_instance.scene_to_duration[i]
                * self.scene_to_actors[i]
                    .union(&standby)
                    .map(|j| self.simplified_instance.actor_to_cost[j])
                    .sum::<i32>();

            successors.push((successor_remaining, weight, i));
        }

        successors
    }

    fn get_base_cost(&self, remaining: &Self::State) -> Option<Self::CostType> {
        if remaining.is_clear() {
            Some(self.single_actor_cost)
        } else {
            None
        }
    }
}

impl Dominance for TalentScheduling {
    type State = FixedBitSet;
    type Key = FixedBitSet;

    fn get_key(&self, state: &Self::State) -> Self::Key {
        state.clone()
    }
}

impl Bound for TalentScheduling {
    type State = FixedBitSet;
    type CostType = i32;

    fn get_dual_bound(&self, remaining: &Self::State) -> Option<Self::CostType> {
        Some(
            remaining
                .ones()
                .map(|i| self.scene_to_base_cost[i])
                .sum::<i32>()
                + self.single_actor_cost,
        )
    }
}

fn main() {
    let timer = Timer::default();
    let args = Args::parse();

    let instance = Instance::read_from_file(&args.input_file).unwrap();
    let ts = TalentScheduling::from(instance);

    let parameters = SearchParameters {
        time_limit: Some(args.time_limit),
        ..Default::default()
    };
    let solution = match args.solver {
        SolverChoice::Cabs => {
            let cabs_parameters = CabsParameters::default();
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = if args.threads.get() > 1 {
                solvers::create_parallel_cabs(
                    ts.clone(),
                    parameters,
                    cabs_parameters,
                    args.threads.get(),
                )
            } else {
                solvers::create_cabs(ts.clone(), parameters, cabs_parameters)
            };
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
        SolverChoice::Astar => {
            println!("Preparing time: {time}s", time = timer.get_elapsed_time());
            let mut solver = solvers::create_astar(ts.clone(), parameters);
            io::run_solver_and_dump_solution_history(&mut solver, &args.history).unwrap()
        }
    };
    io::print_solution_statistics(&solution);

    if let Some(cost) = solution.cost {
        let scenes = ts.reconstruct_solution(&solution.transitions);
        let transitions = scenes
            .iter()
            .map(|t| format!("{t}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("Schedule: {transitions}");

        if ts.instance.validate(&scenes, cost) {
            println!("The solution is valid.");
        } else {
            println!("The solution is invalid.");
        }
    }
}
