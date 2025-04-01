use std::collections::{HashMap, HashSet, VecDeque};

use wasm_bindgen::prelude::*;

mod common;
mod verifier;

use common::*;

#[wasm_bindgen]
pub fn greedy_snake_move_barriers(pos: &[i32], food: &[i32], barriers: &[i32]) -> i32 {
    let head = Point::new(pos[0], pos[1]);
    let mut board = make_board::<8>(food, barriers);
    let init_dir = match pos[0] - pos[2] {
        1 => Direction::R,
        -1 => Direction::L,
        _ => match pos[1] - pos[3] {
            1 => Direction::U,
            -1 => Direction::D,
            _ => unreachable!(),
        },
    };
    if !reachable(&mut board, &head, init_dir) {
        return -1;
    }
    // find the nearest food
    let mut foods = vec![];
    for i in 1..=8 {
        for j in 1..=8 {
            let point = Point::new(i, j);
            if let Some(Tile::Food(depth)) = board.get_tile(point) {
                foods.push((point, depth));
            }
        }
    }
    foods.sort_by(|a, b| a.1.cmp(&b.1));

    if foods.is_empty() {
        panic!("No food found");
    }

    let path = pathfinding(&mut board, &head, init_dir, &foods[0].0);
    if let Some(dir) = path {
        return match dir {
            Direction::U => 0,
            Direction::L => 1,
            Direction::D => 2,
            Direction::R => 3,
        };
    }

    // choose random available direction
    for &dir in &[Direction::U, Direction::L, Direction::D, Direction::R] {
        if matches!(
            (init_dir, dir),
            (Direction::U, Direction::D)
                | (Direction::D, Direction::U)
                | (Direction::L, Direction::R)
                | (Direction::R, Direction::L)
        ) {
            continue; // Snake can't go back
        }
        let next = head.step(dir);
        if board.inbounds(next) {
            if let Some(tile) = board.get_tile(next) {
                match tile {
                    Tile::Empty(_) | Tile::Food(_) => {
                        return match dir {
                            Direction::U => 0,
                            Direction::L => 1,
                            Direction::D => 2,
                            Direction::R => 3,
                        };
                    }
                    _ => {}
                }
            }
        }
    }

        
    panic!("No valid move found")
}

fn make_board<const N: usize>(food: &[i32], barriers: &[i32]) -> Board<N> {
    let mut board = Board::<N>::new();
    food.chunks(2).for_each(|chunk| {
        board.set_tile(chunk, Tile::Food(usize::MAX));
    });
    barriers.chunks(2).for_each(|chunk| {
        board.set_tile(chunk, Tile::Barrier);
    });
    board
}

fn make_board_printing<const N: usize>(pos: &[i32], food: &[i32], barriers: &[i32]) -> Board<N> {
    let mut board = Board::<N>::new();
    let head = Point::new(pos[0], pos[1]);
    board.set_tile(head, Tile::SnakeHead);
    pos.chunks(2).skip(1).for_each(|chunk| {
        board.set_tile(chunk, Tile::SnakeBody);
    });
    food.chunks(2).for_each(|chunk| {
        board.set_tile(chunk, Tile::Food(usize::MAX));
    });
    barriers.chunks(2).for_each(|chunk| {
        board.set_tile(chunk, Tile::Barrier);
    });
    board
}

fn reachable<const N: usize>(board: &mut Board<N>, start: &Point, init_dir: Direction) -> bool {
    use std::collections::VecDeque;
    let mut queue = VecDeque::new();
    let mut visited = HashMap::new();
    queue.push_back((*start, init_dir, 0));
    visited.insert(*start, DirectionMarker::empty());

    while let Some((current, first_dir, depth)) = queue.pop_front() {
        for &dir in &[Direction::U, Direction::L, Direction::D, Direction::R] {
            match (first_dir, dir) {
                (Direction::U, Direction::D)
                | (Direction::D, Direction::U)
                | (Direction::L, Direction::R)
                | (Direction::R, Direction::L) => {
                    continue; // Snake can't go back
                }
                _ => {}
            }
            let next = current.step(dir);
            if !board.inbounds(next) {
                continue;
            }
            if let Some(v) = visited.get_mut(&next) {
                if v.contains(dir) {
                    continue;
                } else {
                    v.set(dir);
                }
            }
            if let Some(tile) = board.get_tile(next) {
                match tile {
                    Tile::Empty(d0) => {
                        visited.insert(next, DirectionMarker::from_dir(dir));
                        queue.push_back((next, dir, depth + 1));
                        if depth < d0 {
                            board.set_tile(next, Tile::Empty(depth + 1));
                        }
                    }
                    Tile::Food(d0) => {
                        visited.insert(next, DirectionMarker::from_dir(dir));
                        queue.push_back((next, dir, depth + 1));
                        if depth < d0 {
                            board.set_tile(next, Tile::Food(depth + 1));
                        }
                    }
                    Tile::Barrier => {
                        continue;
                    }
                    Tile::SnakeHead | Tile::SnakeBody => {
                        unreachable!();
                    }
                }
            }
        }
    }

    let mut dead_end_food_count = 0;
    for i in 1..=N {
        for j in 1..=N {
            let point = Point::new(i as i32, j as i32);
            if let Some(Tile::Food(_)) = board.get_tile(point) {
                if !visited.contains_key(&point) {
                    return false;
                }
                if visited[&point].count() == 0 {
                    dead_end_food_count += 1;
                }
            }
        }
    }
    if dead_end_food_count > 1 {
        return false;
    }
    true
}

fn pathfinding<const N: usize>(board: &mut Board<N>, source: &Point, init_dir: Direction, target: &Point) -> Option<Direction> {

    // Basic BFS setup
    let mut queue: VecDeque<Point> = VecDeque::new();
    let mut visited: HashSet<Point> = HashSet::new();
    // Stores predecessor information to reconstruct the path:
    // Key: Point reached
    // Value: (Point it came from, Direction taken from predecessor to reach key)
    let mut predecessor: HashMap<Point, (Point, Direction)> = HashMap::new();

    // Handle edge case: source is the same as target. No path needed.
    if source == target {
        return None; // Or define behavior based on game rules.
    }

    // Initialize BFS starting from the source point.
    queue.push_back(*source);
    visited.insert(*source);

    // Main BFS loop
    while let Some(current_point) = queue.pop_front() {

        // Explore neighbors in all four directions
        for &dir in &[Direction::U, Direction::L, Direction::D, Direction::R] {

            // --- Constraint Check: Initial Move ---
            // If we are considering the first step from the source,
            // ensure it's not the direct opposite of the snake's current direction.
            if current_point == *source {
                let opposite_init_dir = match init_dir {
                    Direction::U => Direction::D,
                    Direction::D => Direction::U,
                    Direction::L => Direction::R,
                    Direction::R => Direction::L,
                };
                if dir == opposite_init_dir {
                    continue; // Skip this direction, invalid first move.
                }
            }

            // Calculate the coordinates of the neighboring point.
            let next_point = current_point.step(dir);

            // --- Validation Checks for `next_point` ---

            // 1. Bounds Check: Is the neighbor within the board?
            if !board.inbounds(next_point) {
                continue; // Out of bounds.
            }

            // 2. Passability Check: Is the neighbor tile traversable?
            //    Only Empty and Food tiles are considered passable.
            //    Barriers, snake body parts (if marked), etc., block the path.
            let is_passable = match board.get_tile(next_point) {
                Some(Tile::Empty(_)) | Some(Tile::Food(_)) => true, // Allow tiles potentially marked with depth by `reachable`
                _ => false, // Impassable tile (Barrier, SnakeBody, SnakeHead, None)
            };
            if !is_passable {
                continue; // Blocked tile.
            }

            // 3. Visited Check: Have we already found a path to this neighbor?
            //    BFS guarantees the first time we visit a node is via a shortest path.
            if visited.contains(&next_point) {
                continue; // Already visited via an equally short or shorter path.
            }

            // --- Valid Neighbor Found ---

            // Mark the neighbor as visited.
            visited.insert(next_point);
            // Record how we reached this neighbor (for path reconstruction).
            predecessor.insert(next_point, (current_point, dir));
            // Add the neighbor to the queue for further exploration.
            queue.push_back(next_point);

            // --- Target Check ---
            // Have we reached the target destination?
            if next_point == *target {
                // Yes! Target found. Now, reconstruct the path backward from the target
                // to the source to find the very first step taken.
                let mut trace = next_point; // Start backtracking from the target.
                loop {
                    // Retrieve the predecessor and the direction used to reach `trace`.
                    // This entry is guaranteed to exist in the `predecessor` map.
                    let (prev_point, step_dir) = predecessor[&trace];

                    // Check if the predecessor is the original source point.
                    if prev_point == *source {
                        // Found the step immediately following the source.
                        // `step_dir` is the first direction of the shortest path.
                        return Some(step_dir);
                    }
                    // Move one step back along the path.
                    trace = prev_point;
                }
                // This loop always terminates because we know a path exists
                // (we just found the target) and the predecessor map correctly
                // traces back to the source.
            }
        }
    }

    // If the queue becomes empty and we haven't returned yet, it means
    // the target is unreachable from the source under the given constraints.
    None
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use crate::verifier::SnakeState;

    use super::*;

    #[test]
    fn fuzz_snake_movement() {
        let mut success_count = 0;
        let mut unreachable_count = 0;
        for _ in 0..10000 {
            let mut steps = 0;
            let mut game = SnakeState::random(4, 5, 12);
            let (pos, food, bar) = game.get_pair_vec();
            std::io::stdout().flush().unwrap();
            println!("New game board:");
            make_board_printing::<8>(&pos, &food, &bar).pretty_print();

            loop {
                let (pos, food, bar) = game.get_pair_vec();
                let dir = greedy_snake_move_barriers(&pos, &food, &bar);
                if dir == -1 {
                    unreachable_count += 1;
                    println!("Unreachable, manually check. Step: {}", steps);
                    break;
                } else {
                    let dir = match dir {
                        0 => Direction::U,
                        1 => Direction::L,
                        2 => Direction::D,
                        3 => Direction::R,
                        _ => unreachable!(),
                    };
                    steps += 1;
                    let r = game.step(dir);
                    // println!("Step: {}", steps);
                    // println!("Direction: {:?}", dir);
                    // make_board_printing::<8>(&pos, &food, &bar).pretty_print();
                    if r.is_ok() {
                        if r.unwrap() {
                            assert!(game.get_pair_vec().1.is_empty());
                            success_count += 1;
                            println!("Game over. Step: {}", steps);
                            break;
                        }
                    } else {
                        println!("Error: {:?}", r);
                        break;
                    }
                    if steps >= 200 {
                        println!("Too many steps, manually check");
                        println!(
                            "board: {:?}",
                            make_board_printing::<8>(&pos, &food, &bar).pretty_print()
                        );
                        println!("Step: {}", steps);
                        break;
                    }
                }
            }
        }
        println!("Success count: {}", success_count);
        println!("Unreachable count: {}", unreachable_count);
    }
}
