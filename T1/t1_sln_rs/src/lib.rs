use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    U = 0,
    L = 1,
    D = 2,
    R = 3,
}

#[wasm_bindgen]
pub fn greedy_snake_move(pos: &[i32], food: &[i32]) -> i32 {
    let snake: Vec<_> = pos.chunks(2).map(|chunk| (chunk[0], chunk[1])).collect();
    let (fx, fy) = (food[0], food[1]);
    let (head_x, head_y) = snake[0];

    let will_not_colide = |dx: i32, dy: i32| -> bool {
        let (nx, ny) = (head_x + dx, head_y + dy);
        if !(1..=8).contains(&nx) || !(1..=8).contains(&ny) {
            return false;
        }
        !snake.iter().skip(1).take(2).any(|&(x, y)| x == nx && y == ny)
    };

    let preferred = match (head_x.cmp(&fx), head_y.cmp(&fy)) {
        (std::cmp::Ordering::Less, _) => Direction::R,
        (std::cmp::Ordering::Greater, _) => Direction::L,
        (_, std::cmp::Ordering::Less) => Direction::U,
        (_, std::cmp::Ordering::Greater) => Direction::D,
        _ => unreachable!(),
    };

    [preferred]
        .into_iter()
        .chain(
            [Direction::U, Direction::L, Direction::D, Direction::R]
                .into_iter()
                .filter(|&d| d != preferred),
        )
        .map(|dir| {
            let (dx, dy) = match dir {
                Direction::U => (0, 1),
                Direction::L => (-1, 0),
                Direction::D => (0, -1),
                Direction::R => (1, 0),
            };
            (dx, dy, dir)
        })
        .filter(|m| will_not_colide(m.0, m.1))
        .collect::<Vec<_>>()
        .first()
        .map(|m| m.2)
        .unwrap_or(Direction::U) as i32
}
