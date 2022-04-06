#ifndef UTILS_H
#define UTILS_H
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

static int zipf(std::mt19937 &gen, double alpha, int n)
{
  static int first = true;      // Static first time flag
  static double c = 0;          // Normalization constant
  static double *sum_probs;     // Pre-calculated sum of probabilities
  double z;                     // Uniform random number (0 < z < 1)
  int zipf_value = 1;               // Computed exponential value to be returned
  int    i;                     // Loop counter
  int low, high, mid;           // Binary-search bounds

  // Compute normalization constant on first call only
  if (first == true)
  {
    for (i=1; i<=n; i++)
      c = c + (1.0 / pow((double) i, alpha));
    c = 1.0 / c;

    sum_probs = (double *)malloc((n+1)*sizeof(*sum_probs));
    sum_probs[0] = 0;
    for (i=1; i<=n; i++) {
      sum_probs[i] = sum_probs[i-1] + c / pow((double) i, alpha);
    }
    first = false;
  }

  // Pull a uniform random number (0 < z < 1)
  z = 0;
  while (z == 0 || z == 1)
  {
    //z = rand_val(0);
    z = double(gen() & (0xfffffff)) / (0xfffffff);
  }

  // Map z to the value
  low = 1, high = n;
  while (low <= high) {
    mid = floor((low+high)/2);
    if (sum_probs[mid] >= z && sum_probs[mid-1] < z) {
      zipf_value = mid;
      break;
    } else if (sum_probs[mid] >= z) {
      high = mid-1;
    } else {
      low = mid+1;
    }
  }

  // Assert that zipf_value is between 1 and N
  //assert((zipf_value >=1) && (zipf_value <= n));

  return(zipf_value);
}

class HashUtil{
    /**
     * @brief mix 3 32-bit values reversibly
     *
     * For every delta with one or two bits set, and the deltas of all three
     * high bits or all three low bits, whether the original value of a,b,c
     * is almost all zero or is uniformly distributed.
     *
     * If mix() is run forward or backward, at least 32 bits in a,b,c
     * have at least 1/4 probability of changing.
     *
     * If mix() is run forward, every bit of c will change between 1/3 and
     * 2/3 of the time.  (Well, 22/100 and 78/100 for some 2-bit deltas.)
     * mix() was built out of 36 single-cycle latency instructions in a 
     * structure that could supported 2x parallelism, like so:
     *    a -= b; 
     *    a -= c; x = (c>>13);
     *    b -= c; a ^= x;
     *    b -= a; x = (a<<8);
     *    c -= a; b ^= x;
     *    c -= b; x = (b>>13);
     *     ...
     *
     * Unfortunately, superscalar Pentiums and Sparcs can't take advantage 
     * of that parallelism.  They've also turned some of those single-cycle
     * latency instructions into multi-cycle latency instructions. Still,
     * this is the fastest good hash I could find. There were about 2^68
     * to choose from. I only looked at a billion or so.
     */
    #define BOBHASH_MIX(a,b,c) \
    { \
      a -= b; a -= c; a ^= (c>>13); \
      b -= c; b -= a; b ^= (a<<8);  \
      c -= a; c -= b; c ^= (b>>13); \
      a -= b; a -= c; a ^= (c>>12); \
      b -= c; b -= a; b ^= (a<<16); \
      c -= a; c -= b; c ^= (b>>5);  \
      a -= b; a -= c; a ^= (c>>3);  \
      b -= c; b -= a; b ^= (a<<10); \
      c -= a; c -= b; c ^= (b>>15); \
    }

    /**
     * Every bit of the key affects every bit of the return value. 
     * Every 1-bit and 2-bit delta achieves avalanche.
     * About 6 * length + 35 instructions.
     *
     * Use for hash table lookup, or anything where one collision in 2^32 is acceptable.
     * Do NOT use for cryptographic purposes.
     */
    public : 

    static uint32_t BobHash32(const void *key, size_t key_size)
    {
        const uint32_t BOBHASH_GOLDEN_RATIO = 0x9e3779b9;
        uint32_t a = BOBHASH_GOLDEN_RATIO;
        uint32_t b = BOBHASH_GOLDEN_RATIO;
        uint32_t c = 0;
        uint32_t length = key_size;

        uint8_t* work_key = (uint8_t*) key;

        /* handle most of the key */
        while (length >= 12)
        {
            a += (work_key[0] + ((uint32_t)work_key[1] << 8) + ((uint32_t)work_key[2] << 16) + ((uint32_t)work_key[3] << 24));
            b += (work_key[4] + ((uint32_t)work_key[5] << 8) + ((uint32_t)work_key[6] << 16) + ((uint32_t)work_key[7] << 24));
            c += (work_key[8] + ((uint32_t)work_key[9] << 8) + ((uint32_t)work_key[10] << 16)+ ((uint32_t)work_key[11] << 24));
            BOBHASH_MIX (a,b,c);
            work_key += 12; 
            length -= 12;
        }

      /* handle the last 11 bytes */
      c += key_size;
      switch (length)
      {
        case 11: c += ((uint32_t)work_key[10] << 24);
        case 10: c += ((uint32_t)work_key[9] << 16);
        case 9 : c += ((uint32_t)work_key[8] << 8);
        case 8 : b += ((uint32_t)work_key[7] << 24);
        case 7 : b += ((uint32_t)work_key[6] << 16);
        case 6 : b += ((uint32_t)work_key[5] << 8);
        case 5 : b += work_key[4];
        case 4 : a += ((uint32_t)work_key[3] << 24);
        case 3 : a += ((uint32_t)work_key[2] << 16);
        case 2 : a += ((uint32_t)work_key[1] << 8);
        case 1 : a += work_key[0];
      }

      BOBHASH_MIX (a,b,c);

      return c & (0x7fffffff); // 32-bit
    }

    static uint32_t MurmurHash32(uint32_t h)
    {
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }

    static uint64_t MurmurHash64(uint64_t h)
    {
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccd;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53;
        h ^= h >> 33;   
        return h;
    }
} ;


class Metric {
  public:
    static double mean_square_error(std::vector<double> &a, std::vector<double> &b) {
      if (a.size() != b.size()) {
        std::cout << "MSE: Vector lengths don't match." << std::endl;
        return 0;
      }
      double t = 0;
      for (uint i = 0; i < a.size(); i++) {
        t += (a[i] - b[i]) * (a[i] - b[i]);
      }
      t /= a.size();
      return t;
    }
    static double absolute_error(std::vector<double> &a, std::vector<double> &b) {
      if (a.size() != b.size()) {
        std::cout << "AE: Vector lengths don't match." << std::endl;
        return 0;
      }
      double t = 0;
      for (uint i = 0; i < a.size(); i++) {
        //t += (a[i] - b[i]) * (a[i] - b[i]);
        t = std::max(t, std::fabs(a[i] - b[i]));
      }
      return t;
    }
};

static double clip(double x, double l, double r) {
  if (x < l) return l;
  if (x > r) return r;
  return x;
}


#endif