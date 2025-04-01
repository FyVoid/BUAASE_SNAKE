use std::collections::HashSet;

use crate::common::*;

use crate::common::Direction;

pub struct SnakeState {
    body: Vec<Point>,
    food: Vec<Point>,
    barriers: Vec<Point>,
}

impl SnakeState {
    pub fn new(snake: &[Point], food: &[Point], barriers: &[Point]) -> Self {
        let body = snake.to_vec();
        let food = food.to_vec();
        let barriers = barriers.to_vec();

        let state = SnakeState {
            body,
            food,
            barriers,
        };

        assert!(state.verify_state(), "Snake state is invalid");

        state
    }

    // pub fn random(snake_len: usize, food_num: usize, bar_num: usize) -> Self {
    //     use rand::rngs::ThreadRng;
    //     use rand::Rng;
    //     fn rand_point(rng: &mut ThreadRng) -> Point {
    //         Point::new(rng.gen_range(1..=8), rng.gen_range(1..=8))
    //     }
    //     let mut rng = rand::thread_rng();

    //     let head = rand_point(&mut rng);
    //     let mut body = vec![head];
    //     let mut last_part = head;
    //     for _ in 0..snake_len - 1 {
    //         let dir = match rng.gen_range(0..4) {
    //             0 => Direction::U,
    //             1 => Direction::D,
    //             2 => Direction::L,
    //             3 => Direction::R,
    //             _ => unreachable!(),
    //         };
    //         let mut new_part = last_part.step(dir);
    //         while body.contains(&new_part) || !new_part.inbounds(8) {
    //             let dir = match rng.gen_range(0..4) {
    //                 0 => Direction::U,
    //                 1 => Direction::D,
    //                 2 => Direction::L,
    //                 3 => Direction::R,
    //                 _ => unreachable!(),
    //             };
    //             new_part = last_part.step(dir);
    //         }
    //         body.push(new_part);
    //         last_part = new_part;
    //     }

    //     let mut food = vec![];
    //     for _ in 0..food_num {
    //         let mut new_food = rand_point(&mut rng);
    //         while body.contains(&new_food) || food.contains(&new_food) {
    //             new_food = rand_point(&mut rng);
    //         }
    //         food.push(new_food);
    //     }

    //     let mut barriers = vec![];
    //     for _ in 0..bar_num {
    //         let mut new_barrier = rand_point(&mut rng);
    //         while body.contains(&new_barrier)
    //             || food.contains(&new_barrier)
    //             || barriers.contains(&new_barrier)
    //         {
    //             new_barrier = rand_point(&mut rng);
    //         }
    //         barriers.push(new_barrier);
    //     }
    //     let state = SnakeState {
    //         body,
    //         food,
    //         barriers,
    //     };
    //     assert!(state.verify_state(), "Snake state is invalid");
    //     state
    // }

    fn verify_state(&self) -> bool {
        let mut occipied_points: HashSet<Point> = HashSet::new();
        // Check if everything is in bounds
        for point in &self.body {
            if !self.inbounds(point) {
                return false;
            }
        }

        for point in &self.food {
            if !self.inbounds(point) {
                return false;
            }
        }

        for point in &self.barriers {
            if !self.inbounds(point) {
                return false;
            }
        }

        // Check if the snake is continuous
        for i in 0..self.body.len() - 2 {
            if !self.body[i].is_adjacent(&self.body[i + 1]) {
                return false; // Non-continuous snake detected
            }
        }

        // Check if anything collides with the others
        for point in &self.body {
            if occipied_points.contains(point) {
                return false; // Collision with body detected
            }
            occipied_points.insert(*point);
        }

        for point in &self.food {
            if occipied_points.contains(point) {
                return false; // Collision with food detected
            }
            occipied_points.insert(*point);
        }

        for point in &self.barriers {
            if occipied_points.contains(point) {
                return false; // Collision with barriers detected
            }
            occipied_points.insert(*point);
        }

        true
    }

    fn verify_step(&self, dir: Direction) -> Result<bool, String> {
        let new_head = self.body[0].step(dir);
        if !self.inbounds(&new_head) {
            return Err("Out of bounds".to_string());
        }
        if self
            .body
            .iter()
            .take(self.body.len() - 1)
            .any(|&p| p == new_head)
        {
            return Err("Collision with body".to_string());
        }
        if self.barriers.contains(&new_head) {
            return Err("Collision with barrier".to_string());
        }
        if self.food.contains(&new_head) {
            return Ok(true);
        }
        Ok(false)
    }

    pub fn step(&mut self, dir: Direction) -> Result<bool, String> {
        let ate = self.verify_step(dir)?;
        let new_head = self.body[0].step(dir);
        self.body.insert(0, new_head);
        self.body.pop();
        if ate {
            self.food.retain(|&p| p != new_head);
        }
        if self.food.is_empty() {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn get_pair_vec(&self) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let body = self.body.iter().map(Point::make_pair).map(|p| [p.0, p.1]).collect::<Vec<_>>().into_iter().flatten().collect::<Vec<i32>>();
        let food = self.food.iter().map(Point::make_pair).map(|p| [p.0, p.1]).collect::<Vec<_>>().into_iter().flatten().collect::<Vec<i32>>();
        let barriers = self.barriers.iter().map(Point::make_pair).map(|p| [p.0, p.1]).collect::<Vec<_>>().into_iter().flatten().collect::<Vec<i32>>();
        (body, food, barriers)
    }

    fn inbounds(&self, point: &Point) -> bool {
        point.inbounds(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake_state() {
        let snake = vec![Point::new(1, 1), Point::new(1, 2), Point::new(1, 3)];
        let food = vec![Point::new(2, 2)];
        let barriers = vec![Point::new(3, 3)];
        let state = SnakeState::new(&snake, &food, &barriers);
        assert!(state.verify_state(), "Snake state is invalid");
    }
}
