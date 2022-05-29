#![allow(dead_code)]
#![warn(missing_docs)]

//! This library contains objects to perform the linear algebra neccesary to create
//! wireframe objects, manipulate, move, and view them in a 3D environment.
//! This library is a work in progress.
use std::ops;
use std::ops::Mul;
use std::ops::{Index, IndexMut};

/// An abstraction over a 2D matrix of `f64`s. the foundation of this library
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Matrix<const ROWS: usize, const COLS: usize> {
    matrix: [[f64; COLS]; ROWS],
}

/// A Matrix with only one column
pub type ColumnVector<const SIZE: usize> = Matrix<SIZE, 1>;

impl<const ROWS: usize, const COLS: usize> Index<(usize, usize)> for Matrix<ROWS, COLS> {
    type Output = f64;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.matrix[index.0][index.1]
    }
}

impl<const ROWS: usize, const COLS: usize> IndexMut<(usize, usize)> for Matrix<ROWS, COLS> {
    ///Mutably refer to a coordinate in a matrix by supplying a tuple (column, row)
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.matrix[index.0][index.1]
    }
}

//From implementation
/// Creates matrix from 2d array of `f64`s
impl<const ROWS: usize, const COLS: usize> From<[[f64; COLS]; ROWS]> for Matrix<ROWS, COLS> {
    fn from(mtx: [[f64; COLS]; ROWS]) -> Self {
        Self { matrix: mtx }
    }
}

/// Performs `Matrix`-`Matrix` multiplication
/// The two Matrices must share one COMMON dimension, where one has the COMMON value for the row number and the other `Matrix` has it for the
/// column number
impl<const COMMON: usize, const LHSROWS: usize, const RHSCOLS: usize> Mul<Matrix<COMMON, RHSCOLS>>
    for Matrix<LHSROWS, COMMON>
{
    type Output = Matrix<LHSROWS, RHSCOLS>;
    fn mul(self, rhs: Matrix<COMMON, RHSCOLS>) -> Self::Output {
        let mut ret = Matrix::<LHSROWS, RHSCOLS>::zero();

        //TODO: Make this less bad
        for i in 0..LHSROWS {
            for j in 0..RHSCOLS {
                let mut dot_product: f64 = 0.0;
                for k in 0..COMMON {
                    dot_product += self[(i, k)] * rhs[(k, j)];
                }
                ret[(i, j)] = dot_product;
            }
        }

        ret
    }
}

//Matrix Constructors
impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
    /// Produces a `Matrix` of ROWS rows and COLS columns populated with all 0.0s
    pub fn zero() -> Self {
        Self {
            matrix: [[0.0; COLS]; ROWS],
        }
    }

    /// Produces a `Matrix` of ROWS rows and COLS columns populated with all 1.0s
    pub fn ones() -> Self {
        Self {
            matrix: [[1.0; COLS]; ROWS],
        }
    }

    /// Construct new `Matrix` using a literal 2d array of `f64`s as input
    pub fn new(mtx: [[f64; COLS]; ROWS]) -> Self {
        Self { matrix: mtx }
    }
}

//For methods requiring square matrices
impl<const ROWS: usize> Matrix<ROWS, ROWS> {
    /// Produces an identity `Matrix` which MUST be square in dimensions
    pub fn identity() -> Self {
        let mut ret = Self::zero();
        for i in 0..ROWS {
            ret[(i, i)] = 1.0;
        }

        ret
    }
}

//Matrix addition with + operator
impl<const ROWS: usize, const COLS: usize> ops::Add<Matrix<ROWS, COLS>> for Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;
    fn add(self, rhs: Self::Output) -> Self::Output {
        let mut ret = Self::zero();

        for ((i, lhs_row), rhs_row) in self.matrix.iter().enumerate().zip(rhs.matrix.iter()) {
            for ((j, lhs_val), rhs_val) in lhs_row.iter().enumerate().zip(rhs_row.iter()) {
                ret[(i, j)] = lhs_val + rhs_val;
            }
        }

        ret
    }
}

//Vector Addition with += operator
impl<const ROWS: usize, const COLS: usize> ops::AddAssign<Matrix<ROWS, COLS>>
    for Matrix<ROWS, COLS>
{
    fn add_assign(&mut self, rhs: Self) {
        //TODO
        todo!();
    }
}

/// To enable `Matrix`-scalar multiplication with * operator
impl<const ROWS: usize, const COLS: usize> ops::Mul<f64> for Matrix<ROWS, COLS> {
    type Output = Matrix<ROWS, COLS>;
    fn mul(self, rhs: f64) -> Self::Output {
        for row in self.matrix.iter() {
            row.map(|x| x * rhs);
        }

        self
    }
}

#[derive(Debug)]
/// Represents a coordinate in 3D space. It is composed of a [`ColumnVector`] of size 3, and a [`Vec`<[`Coordinate3d`]>] to record its edges.
pub struct Coordinate3d {
    pos: ColumnVector<3>,
    edges: Vec<Coordinate3d>,
}

impl Coordinate3d {
    /// Translates a `Coordinate3d` by a `ColumnVector` quantity
    fn translate(&mut self, vector: ColumnVector<3>) {
        self.pos += vector;
    }

    /// Calculates the distance between two `Coordinate3d`'s
    fn distance(self, cord: Coordinate3d) -> f64 {
        //pythagorean theorem
        //sqrt((x1 + x2)^2 + (y1 + y2)^2 + (z1 + z2)^2)
        let mut sum_of_squares: f64 = 0.0;
        for axis in 0..3 {
            sum_of_squares += (self.pos[(axis, 1)] + cord.pos[(axis, 1)]).powi(2);
        }

        sum_of_squares.sqrt()
    }
}

#[derive(Debug)]
/// A collection of `Coordinate3d` verticies (which have edges), and an arbitrary origin. Used to represent 3D wireframe models
pub struct Mesh {
    vertices: Vec<Coordinate3d>,
    origin: Coordinate3d,
}

impl Mesh {
    /// Translate a mesh by a vector quantity. This function is implemented by calling
    /// Coordinate3d::translate for each vertex in the mesh.
    pub fn translate(&mut self, vector: ColumnVector<3>) {
        for vertex in self.vertices.iter_mut() {
            vertex.translate(vector);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_ones() {
        assert_eq!(
            Matrix::<3, 4>::ones().matrix,
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0]
            ]
        );
    }

    #[test]
    fn constructor_zero() {
        assert_eq!(
            Matrix::<3, 4>::zero().matrix,
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ]
        );
    }

    #[test]
    fn constructor_identity() {
        assert_eq!(
            Matrix::<4, 4>::identity().matrix,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        );
    }

    #[test]
    fn matrix_from() {
        let input: [[f64; 2]; 3] = [[0.1066, 1066.0], [99.9, 100.0], [12.3, 45.6]];
        let expected = Matrix { matrix: input };
        assert_eq!(Matrix::<3, 2>::new(input), expected);
    }

    #[test]
    //this test should not use From() or Into() because it will make the result and expected
    //created by the same function call and it will always pass even on bad input.
    fn constructor_literal() {
        let input = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        assert_eq!(Matrix::<3, 3>::new(input).matrix, input);
    }

    #[test]
    fn matrix_index() {
        let mtx: Matrix<3, 3> = Matrix::<3, 3>::identity();
        assert_eq!(mtx[(0, 0)], 1.0);
        assert_eq!(mtx[(2, 1)], 0.0);
    }

    #[test]
    fn matrix_index_mut() {
        let mut mtx: Matrix<3, 3> = Matrix::<3, 3>::zero();
        assert_eq!(mtx[(1, 2)], 0.0);
        mtx[(1, 2)] = 10.66;
        assert_eq!(mtx[(1, 2)], 10.66);
    }

    #[test]
    fn matrix_add() {
        assert_eq!(
            (Matrix::<2, 2>::ones() + Matrix::<2, 2>::ones()).matrix,
            [[2.0, 2.0], [2.0, 2.0]]
        );
    }

    #[test]
    fn matrix_mul_by_matrix() {
        let identity = Matrix::<3, 3>::identity();
        let expected = Matrix::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!((identity * identity), expected);
    }

    #[test]
    fn matrix_mul_by_matrix_2() {
        let mtx1 = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let mtx2 = Matrix::<3, 2>::new([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
        let expected = Matrix::from([[58.0, 64.0], [139.0, 154.0]]);
        let product = mtx1 * mtx2;

        assert_eq!(product, expected);
    }

    #[test]
    fn matrix_mul_by_matrix_3() {
        let mtx1 = Matrix::<1, 2>::new([[2.0, 3.0]]);
        let mtx2 = Matrix::<2, 3>::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let expected = Matrix::from([[14.0, 19.0, 24.0]]);
        let product = mtx1 * mtx2;

        assert_eq!(Matrix::from(product), expected);
    }
}
