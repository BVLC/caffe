/* adapted from http://en.wikipedia.org/wiki/Boyer–Moore_string_search_algorithm */
#ifndef BOYER_MOORE_H
#define BOYER_MOORE_H

#include <stdint.h>
#include <limits.h>
#include <string.h>

#define ALPHABET_SIZE (1 << CHAR_BIT)

static void compute_prefix(const uint8_t* str, size_t size, int result[]) {
	size_t q;
	int k;
	result[0] = 0;

	k = 0;
	for (q = 1; q < size; q++) {
		while (k > 0 && str[k] != str[q])
			k = result[k-1];

		if (str[k] == str[q])
			k++;

		result[q] = k;
	}
}

static void prepare_badcharacter_heuristic(const uint8_t *str, size_t size, int result[ALPHABET_SIZE]) {
	size_t i;

	for (i = 0; i < ALPHABET_SIZE; i++)
		result[i] = -1;

	for (i = 0; i < size; i++)
		result[str[i]] = i;
}

void prepare_goodsuffix_heuristic(const uint8_t *normal, const size_t size, int result[]) {
	const uint8_t *left = normal;
	const uint8_t *right = left + size;
	uint8_t * reversed = new uint8_t[size+1];
	uint8_t *tmp = reversed + size;
	size_t i;

	/* reverse string */
	*tmp = 0;
	while (left < right)
		*(--tmp) = *(left++);

	int * prefix_normal = new int[size];
	int * prefix_reversed = new int[size];

	compute_prefix(normal, size, prefix_normal);
	compute_prefix(reversed, size, prefix_reversed);

	for (i = 0; i <= size; i++) {
		result[i] = size - prefix_normal[size-1];
	}

	for (i = 0; i < size; i++) {
		const int j = size - prefix_reversed[i];
		const int k = i - prefix_reversed[i]+1;

		if (result[j] > k)
			result[j] = k;
	}

	delete[] reversed;
	delete[] prefix_normal;
	delete[] prefix_reversed;
}

/*
* Boyer-Moore search algorithm
*/
const uint8_t *boyermoore_search(const uint8_t *haystack, size_t haystack_len, const uint8_t *needle, size_t needle_len) {
	/*
	* Simple checks
	*/
	if(haystack_len == 0)
		return NULL;
	if(needle_len == 0)
		return NULL;
	if(needle_len > haystack_len)
		return NULL;

	/*
	* Initialize heuristics
	*/
	int badcharacter[ALPHABET_SIZE];
	int * goodsuffix = new int[needle_len+1];

	prepare_badcharacter_heuristic(needle, needle_len, badcharacter);
	prepare_goodsuffix_heuristic(needle, needle_len, goodsuffix);

	/*
	* Boyer-Moore search
	*/
	size_t s = 0;
	while(s <= (haystack_len - needle_len))
	{
		size_t j = needle_len;
		while(j > 0 && needle[j-1] == haystack[s+j-1])
			j--;

		if(j > 0)
		{
			int k = badcharacter[haystack[s+j-1]];
			int m;
			if(k < (int)j && (m = j-k-1) > goodsuffix[j])
				s+= m;
			else
				s+= goodsuffix[j];
		}
		else
		{
			delete[] goodsuffix;
			return haystack + s;
		}
	}

	delete[] goodsuffix;
	/* not found */
	return NULL;
}

#endif	/* BoyerMoore.h */
