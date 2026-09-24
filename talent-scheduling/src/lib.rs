use clap::{Parser, ValueEnum};
use fixedbitset::FixedBitSet;
use itertools::Itertools;
use rpid::io;
use std::error::Error;
use std::fs;

#[derive(Clone, Debug)]
pub struct Instance {
    pub actor_to_scenes: Vec<Vec<usize>>,
    pub actor_to_cost: Vec<i32>,
    pub scene_to_duration: Vec<i32>,
}

impl Instance {
    pub fn read_from_file(filename: &str) -> Result<Self, Box<dyn Error>> {
        let file = fs::read_to_string(filename)?;
        let mut digits = file.split_whitespace();

        digits.next().ok_or("empty file".to_owned())?;
        let n = digits
            .next()
            .ok_or("failed to parse the number of scenes".to_owned())?
            .parse()?;
        let m = digits
            .next()
            .ok_or("failed to parse the number of actors")?
            .parse()?;
        let matrix: Vec<Vec<i32>> = io::read_matrix(&mut digits, m, n + 1)?;
        let actor_to_scenes = matrix
            .iter()
            .map(|row| (0..n).filter(|&i| row[i] == 1).collect())
            .collect();
        let actor_to_cost = matrix.iter().map(|row| row[n]).collect();
        let scene_to_duration = io::read_vector(&mut digits, n)?;

        Ok(Self {
            actor_to_scenes,
            actor_to_cost,
            scene_to_duration,
        })
    }

    pub fn validate(&self, scenes: &[usize], cost: i32) -> bool {
        if scenes.len() != self.scene_to_duration.len() {
            println!("Invalid number of scenes: {len}", len = scenes.len());

            return false;
        }

        let m = self.actor_to_cost.len();
        let scene_to_actors = self.create_scene_to_actors();
        let mut on_location_actors = FixedBitSet::with_capacity(m);
        let mut shot = vec![false; self.scene_to_duration.len()];
        let mut recomputed_cost = 0;

        for (i, &scene) in scenes.iter().enumerate() {
            if scene >= self.scene_to_duration.len() {
                println!("Invalid scene index: {scene}");

                return false;
            }

            if shot[scene] {
                println!("Scene {scene} is shot twice");

                return false;
            }

            on_location_actors.union_with(&scene_to_actors[scene]);

            recomputed_cost += self.scene_to_duration[scene]
                * self
                    .actor_to_cost
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| on_location_actors.contains(i))
                    .map(|(_, &cost)| cost)
                    .sum::<i32>();

            let mut working_actors = FixedBitSet::with_capacity(m);
            scenes[i + 1..].iter().for_each(|&scene| {
                working_actors.union_with(&scene_to_actors[scene]);
            });
            on_location_actors.intersect_with(&working_actors);

            shot[scene] = true;
        }

        if recomputed_cost != cost {
            println!("Invalid cost: {cost} != {recomputed_cost}");

            return false;
        }

        true
    }

    pub fn create_scene_to_actors(&self) -> Vec<FixedBitSet> {
        let m = self.actor_to_cost.len();
        let n = self.scene_to_duration.len();
        let mut scene_to_actors = (0..n)
            .map(|_| FixedBitSet::with_capacity(m))
            .collect::<Vec<_>>();

        for (actor, scenes) in self.actor_to_scenes.iter().enumerate() {
            for &scene in scenes {
                scene_to_actors[scene].set(actor, true);
            }
        }

        scene_to_actors
    }

    fn eliminate_single_scene_actors(&self) -> Option<(Self, i32)> {
        let mut single_actor_cost = 0;
        let mut keep = Vec::with_capacity(self.actor_to_cost.len());
        let mut updated = false;

        for (i, scenes) in self.actor_to_scenes.iter().enumerate() {
            match scenes.len() {
                1 => {
                    single_actor_cost += self.actor_to_cost[i] * self.scene_to_duration[scenes[0]];

                    if !updated {
                        updated = true;
                    }
                }
                len if len > 1 => keep.push(i),
                _ => {
                    updated = true;
                }
            }
        }

        if updated {
            let actor_to_scenes = keep
                .iter()
                .map(|&i| self.actor_to_scenes[i].clone())
                .collect();
            let actor_to_cost = keep.iter().map(|&i| self.actor_to_cost[i]).collect();

            Some((
                Self {
                    actor_to_scenes,
                    actor_to_cost,
                    scene_to_duration: self.scene_to_duration.clone(),
                },
                single_actor_cost,
            ))
        } else {
            None
        }
    }

    fn concatenate_duplicate_scenes(&self) -> Option<(Self, Vec<Vec<usize>>)> {
        let n = self.scene_to_duration.len();
        let mut scene_to_new_scene = (0..n).collect::<Vec<_>>();
        let mut new_scene_to_scenes = Vec::with_capacity(n);
        let scene_to_actors = self.create_scene_to_actors();
        let mut index = 0;
        let mut merged = vec![false; n];
        let mut new_scene_to_duration = Vec::new();
        let mut updated = false;

        for i in 0..n {
            if merged[i] {
                continue;
            }

            new_scene_to_duration.push(self.scene_to_duration[i]);
            scene_to_new_scene[i] = index;
            new_scene_to_scenes.push(vec![i]);

            for j in (i + 1)..n {
                if scene_to_actors[i] == scene_to_actors[j] {
                    new_scene_to_scenes[index].push(j);
                    scene_to_new_scene[j] = index;
                    new_scene_to_duration[index] += self.scene_to_duration[j];
                    merged[j] = true;

                    if !updated {
                        updated = true;
                    }
                }
            }

            index += 1;
        }

        if updated {
            let actor_to_scenes = self
                .actor_to_scenes
                .iter()
                .map(|scenes| {
                    scenes
                        .iter()
                        .map(|&scene| scene_to_new_scene[scene])
                        .unique()
                        .collect()
                })
                .collect();
            let scene_to_duration = new_scene_to_duration;

            Some((
                Self {
                    actor_to_scenes,
                    actor_to_cost: self.actor_to_cost.clone(),
                    scene_to_duration,
                },
                new_scene_to_scenes,
            ))
        } else {
            None
        }
    }

    pub fn simplify(&self) -> (Self, i32, Vec<Vec<usize>>) {
        let mut instance = self.clone();
        let mut single_actor_cost = 0;
        let mut scene_to_originals = (0..self.scene_to_duration.len())
            .map(|i| vec![i])
            .collect::<Vec<_>>();

        loop {
            let mut updated = false;

            if let Some((new_instance, cost)) = instance.eliminate_single_scene_actors() {
                instance = new_instance;
                single_actor_cost += cost;
                updated = true;
            }

            if let Some((new_instance, new_scene_to_scenes)) =
                instance.concatenate_duplicate_scenes()
            {
                instance = new_instance;
                scene_to_originals = new_scene_to_scenes
                    .into_iter()
                    .map(|scenes| {
                        scenes
                            .into_iter()
                            .flat_map(|i| scene_to_originals[i].iter().cloned())
                            .collect()
                    })
                    .collect();
                updated = true;
            }

            if !updated {
                break;
            }
        }

        (instance, single_actor_cost, scene_to_originals)
    }
}

#[derive(Debug, Clone, ValueEnum)]
pub enum SolverChoice {
    Cabs,
    Astar,
}

#[derive(Debug, Parser)]
pub struct Args {
    #[arg(
        short = 'j',
        long,
        default_value = "1",
        help = "Number of threads for CABS"
    )]
    pub threads: std::num::NonZeroUsize,
    #[arg(help = "Input file")]
    pub input_file: String,
    #[arg(short, long, value_enum, default_value_t = SolverChoice::Cabs, help = "Solver")]
    pub solver: SolverChoice,
    #[arg(long, default_value_t = String::from("history.csv"), help = "File to save the history")]
    pub history: String,
    #[arg(short, long, default_value_t = 1800.0, help = "Time limit")]
    pub time_limit: f64,
}
