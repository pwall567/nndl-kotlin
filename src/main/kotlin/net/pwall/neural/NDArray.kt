/*
 * @(#) NDArray.kt
 *
 * nndl-kotlin Neural Networks and Deep Learning - Kotlin
 * Copyright (c) 2018 Peter Wall
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package net.pwall.neural

import java.util.Arrays
import java.util.Random

/**
 * A class to represent a 2-dimensional array for the neural network implementation.
 *
 * The name `NDArray` comes from the "numpy" class which performs this function in the original python code.  In this
 * case it's something of a misnomer since the name was supposed to refer to an n-dimensional array, but this
 * implementation only handles 2-dimensional arrays.
 *
 * For performance reasons, the array (of [Double]) is stored as a flat array and the [get] and [set] operations
 * calculate the actual offset into the array.
 *
 * @author  Peter Wall
 */
class NDArray(val dim1: Int, val dim2 : Int, initArray: DoubleArray? = null) {

    private val array = initArray ?: DoubleArray(dim1 * dim2)

    init {
        if (array.size != dim1 * dim2)
            throw IllegalArgumentException("Array is incorrect size")
    }

    operator fun get(i1: Int, i2: Int) = array[i1 * dim2 + i2]

    operator fun set(i1: Int, i2: Int, value: Double) {
        array[i1 * dim2 + i2] = value
    }

    operator fun plus(other: NDArray) : NDArray {
        if (dim1 != other.dim1 || dim2 != other.dim2)
            throw IllegalArgumentException("Arrays must be same dimensions")
        return NDArray(dim1, dim2, DoubleArray(dim1 * dim2) { i -> array[i] + other.array[i] })
    }

    operator fun plusAssign(other: NDArray) {
        if (dim1 != other.dim1 || dim2 != other.dim2)
            throw IllegalArgumentException("Arrays must be same dimensions")
        for (i in array.indices)
            array[i] += other.array[i]
    }

    operator fun minus(other: NDArray) : NDArray {
        if (dim1 != other.dim1 || dim2 != other.dim2)
            throw IllegalArgumentException("Arrays must be same dimensions")
        return NDArray(dim1, dim2, DoubleArray(dim1 * dim2) { i -> array[i] - other.array[i] })
    }

    operator fun minusAssign(other: NDArray) {
        if (dim1 != other.dim1 || dim2 != other.dim2)
            throw IllegalArgumentException("Arrays must be same dimensions")
        for (i in array.indices)
            array[i] -= other.array[i]
    }

    operator fun times(other: NDArray) : NDArray {
        if (dim1 != other.dim1 || dim2 != other.dim2)
            throw IllegalArgumentException("Arrays must be same dimensions")
        return NDArray(dim1, dim2, DoubleArray(dim1 * dim2) { i -> array[i] * other.array[i] })
    }

    operator fun times(value: Double) = NDArray(dim1, dim2, DoubleArray(dim1 * dim2) { i -> array[i] * value })

    operator fun timesAssign(other: NDArray) {
        if (dim1 != other.dim1 || dim2 != other.dim2)
            throw IllegalArgumentException("Arrays must be same dimensions")
        for (i in array.indices)
            array[i] *= other.array[i]
    }

    /**
     * Matrix multiplication of two [NDArray]s.  Do a web search for "matrix multiplication" for an explanation.
     * The second dimension of the first array must equal the first dimension of the second array.
     *
     * @param   other   the other [NDArray]
     * @return          the matrix product
     * @throws  IllegalArgumentException if the arrays are of incompatible dimensions
     */
    infix fun dot(other: NDArray) : NDArray {
        if (dim2 != other.dim1)
            throw IllegalArgumentException("Array dimensions not compatible")
        val newArray = DoubleArray(dim1 * other.dim2)
        for (i in 0 until dim1) {
            for (j in 0 until other.dim2) {
                var sum = 0.0
                for (k in 0 until dim2)
                    sum += this[i, k] * other[k, j]
                newArray[i * other.dim2 + j] = sum
            }
        }
        return NDArray(dim1, other.dim2, newArray)
    }

    /**
     * Initialise the array with Gaussian distributed values (mean `0.0`, standard deviation `1.0`).  The [Random] used
     * as the source of the values may be supplied as an argument to allow the user to use a [Random] with a known seed
     * for repeatable results.
     *
     * @param   r       the [Random] (default is a new [Random])
     */
    fun init(r: Random = Random()) {
        for (i in array.indices)
            array[i] = r.nextGaussian()
    }

    /**
     * Transpose the two dimensions of the array, so that an array of dimension x,y becomes y,x.
     *
     * Note that when either dimension is `1`, the layout of the actual array is unchanged so this case is optimised.
     *
     * @return  a new [NDArray] with the dimensions transposed
     */
    fun transpose() : NDArray {
        if (dim1 == 1 || dim2 == 1)
            return NDArray(dim2, dim1, array.copyOf()) // optimisation - array layout is the same
        return NDArray(dim2, dim1, DoubleArray(dim1 * dim2) { i -> get(i % dim1, i / dim1) })
    }

    /**
     * Apply a function to each [Double] in the array, creating a new array.
     *
     * @param   function    a function that takes a [Double] and returns a [Double]
     * @return  the new [NDArray]
     */
    fun apply(function: (f: Double) -> Double) : NDArray {
        return NDArray(dim1, dim2, DoubleArray(dim1 * dim2) { i -> function(array[i]) })
    }

    fun copy() = NDArray(dim1, dim2, array.copyOf())

    override fun equals(other: Any?): Boolean {
        if (other != null && other is NDArray) {
            if (dim1 != other.dim1 || dim2 != other.dim2)
                return false
            for (i in array.indices)
                if (array[i] != other.array[i])
                    return false
            return true
        }
        return false
    }

    override fun hashCode(): Int {
        var result = dim1
        result = 31 * result + dim2
        result = 31 * result + Arrays.hashCode(array)
        return result
    }

    companion object {

        fun fromDoubleArray(doubleArray: DoubleArray) : NDArray = NDArray(doubleArray.size, 1, doubleArray)

    }

}
