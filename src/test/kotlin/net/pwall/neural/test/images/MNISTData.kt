/*
 * @(#) MNISTData.kt
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

package net.pwall.neural.test.images

import java.io.EOFException
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream

open class MNISTData(filename: String, magicNumber: Int) {

    private val input: InputStream = FileInputStream(filename)

    init {
        if (read32() != magicNumber)
            throw IOException("Incorrect magic number")
    }

    fun read8() : Int {
        val result = input.read()
        if (result < 0)
            throw EOFException("Unexpected EOF")
        return result
    }

    fun read32() : Int {
        var result = 0
        for (i in 0..3)
            result = (result shl 8) or (read8() and 0xFF)
        return result
    }

    fun readArray(buf: ByteArray, len: Int) {
        var length = len
        var offset = 0
        while (length > 0) {
            val n = input.read(buf, offset, length)
            if (n < 0)
                throw EOFException("Unexpected EOF")
            offset += n
            length -= n
        }
    }

}
