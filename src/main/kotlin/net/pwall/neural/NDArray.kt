/*
 * @(#) NDArray.java
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

import java.util.*

class NDArray(private val dim1: Int, private val dim2 : Int, initArray: DoubleArray? = null) {

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

    infix fun dot(other: NDArray) : NDArray {
        if (dim2 != other.dim1)
            throw IllegalArgumentException("Array dimensions not compatible")
        return NDArray(dim1, other.dim2, DoubleArray(dim1 * other.dim2, fun (x: Int) : Double {
            var sum = 0.0
            for (k in 0 until dim2)
                sum += get(x / other.dim2, k) * other[k, x % other.dim2]
            return sum
        }))
    }

    fun init(r: Random) {
        for (i in array.indices)
            array[i] = r.nextGaussian()
    }

    fun transpose() : NDArray {
        if (dim1 == 1 || dim2 == 1)
            return NDArray(dim2, dim1, array)
        return NDArray(dim2, dim1, DoubleArray(dim1 * dim2) { i -> get(i % dim1, i / dim1) })
    }

    fun apply(function: (f: Double) -> Double) : NDArray {
        val newArray = DoubleArray(dim1 * dim2) { i -> function(array[i]) }
        return NDArray(dim1, dim2, newArray)
    }

    fun copy() = NDArray(dim1, dim2, DoubleArray(dim1 * dim2) { i -> array[i] })

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
