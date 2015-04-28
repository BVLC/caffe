#include "caffe/util/permutohedral.hpp"

#ifdef WIN32
static float round( float v ) {
  return floor( v+0.5f );
}
#endif

/************************************************/
/***                Hash Table                ***/
/************************************************/

class HashTable{
protected:
  size_t key_size_, filled_, capacity_;
  short * keys_;
  int * table_;
  void grow(){
    // Swap out the old memory
    short * old_keys = keys_;
    int * old_table = table_;
    size_t old_capacity = capacity_;
    capacity_ *= 2;
    // Allocate the new memory
    keys_ = new short[ (old_capacity+10)*key_size_ ];
    table_ = new int[ capacity_ ];
    memset( table_, -1, capacity_*sizeof(int) );
    memcpy( keys_, old_keys, filled_*key_size_*sizeof(short) );
		
    // Reinsert each element
    for( size_t i=0; i<old_capacity; i++ )
      if (old_table[i] >= 0){
	int e = old_table[i];
	size_t h = hash( old_keys+(getKey(e)-keys_) ) % capacity_;
	for(; table_[h] >= 0; h = h<capacity_-1 ? h+1 : 0);
	table_[h] = e;
      }
		
    delete [] old_keys;
    delete [] old_table;
  }
  size_t hash( const short * k ) {
    size_t r = 0;
    for( size_t i=0; i<key_size_; i++ ){
      r += k[i];
      r *= 1664525;
    }
    return r;
  }
 public:
  explicit HashTable( int key_size, int n_elements ) : key_size_ ( key_size ), filled_(0), capacity_(2*n_elements) {
    table_ = new int[ capacity_ ];
    keys_ = new short[ (capacity_/2+10)*key_size_ ];
    memset( table_, -1, capacity_*sizeof(int) );
  }
  ~HashTable() {
    delete [] keys_;
    delete [] table_;
  }
  int size() const {
    return (int)filled_;
  }
  void reset() {
    filled_ = 0;
    memset( table_, -1, capacity_*sizeof(int) );
  }
  int find( const short * k, bool create = false ){
    if (2*filled_ >= capacity_) grow();
    // Get the hash value
    size_t h = hash( k ) % capacity_;
    // Find the element with he right key, using linear probing
    while(1){
      int e = table_[h];
      if (e==-1){
	if (create){
	  // Insert a new key and return the new id
	  for( size_t i=0; i<key_size_; i++ )
	    keys_[ filled_*key_size_+i ] = k[i];
	  return table_[h] = (int)filled_++;
	}
	else
	  return -1;
      }
      // Check if the current key is The One
      bool good = true;
      for( size_t i=0; i<key_size_ && good; i++ )
	if (keys_[ e*key_size_+i ] != k[i])
	  good = false;
      if (good)
	return e;
      // Continue searching
      h++;
      if (h==capacity_) h = 0;
    }
  }
  const short * getKey( int i ) const{
    return keys_+i*key_size_;
  }

};

/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/
Permutohedral::Permutohedral() 
  : offset_( NULL ),barycentric_( NULL ),blur_neighbors_( NULL ),N_ ( 0 ),M_ ( 0 ),d_ ( 0 ) {
}

Permutohedral::~Permutohedral() {
  if (barycentric_)    delete[] barycentric_;
  if (offset_)         delete[] offset_;
  if (blur_neighbors_) delete[] blur_neighbors_;
}

void Permutohedral::init(const float* feature, int feature_size, int N) {
#ifdef SSE_PERMUTOHEDRAL
    // Compute the lattice coordinates for each feature [there is going to be a lot of magic here
    N_ = N;
    d_ = feature_size;
    HashTable hash_table( d_, N_/**(d_+1)*/ );
		
    const int blocksize = sizeof(__m128) / sizeof(float);
    const __m128 invdplus1   = _mm_set1_ps( 1.0f / (d_+1) );
    const __m128 dplus1      = _mm_set1_ps( d_+1 );
    const __m128 Zero        = _mm_set1_ps( 0 );
    const __m128 One         = _mm_set1_ps( 1 );

    // Allocate the class memory
    if (offset_) delete [] offset_;
    offset_ = new int[ (d_+1)*(N_+16) ];
    memset( offset_, 0, (d_+1)*(N_+16)*sizeof(int) );
		
    if (barycentric_) delete [] barycentric_;
    barycentric_ = new float[ (d_+1)*(N_+16) ];
    memset( barycentric_, 0, (d_+1)*(N_+16)*sizeof(float) );
		
    // Allocate the local memory
    __m128 * scale_factor = (__m128*) _mm_malloc( (d_  )*sizeof(__m128) , 16 );
    __m128 * f            = (__m128*) _mm_malloc( (d_  )*sizeof(__m128) , 16 );
    __m128 * elevated     = (__m128*) _mm_malloc( (d_+1)*sizeof(__m128) , 16 );
    __m128 * rem0         = (__m128*) _mm_malloc( (d_+1)*sizeof(__m128) , 16 );
    __m128 * rank         = (__m128*) _mm_malloc( (d_+1)*sizeof(__m128), 16 );
    float * barycentric = new float[(d_+2)*blocksize];
    short * canonical = new short[(d_+1)*(d_+1)];
    short * key = new short[d_+1];
		
    // Compute the canonical simplex
    for( int i=0; i<=d_; i++ ){
      for( int j=0; j<=d_-i; j++ )
	canonical[i*(d_+1)+j] = i;
      for( int j=d_-i+1; j<=d_; j++ )
	canonical[i*(d_+1)+j] = i - (d_+1);
    }
		
    // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
    float inv_std_dev = sqrt(2.0 / 3.0)*(d_+1);
    // Compute the diagonal part of E (p.5 in [Adams etal 2010])
    for( int i=0; i<d_; i++ )
      scale_factor[i] = _mm_set1_ps( 1.0 / sqrt( (i+2)*(i+1) ) * inv_std_dev );
		
    // Setup the SSE rounding
#ifndef __SSE4_1__
    const unsigned int old_rounding = _mm_getcsr();
    _mm_setcsr( (old_rounding&~_MM_ROUND_MASK) | _MM_ROUND_NEAREST );
#endif

    // Compute the simplex each feature lies in
    for( int k=0; k<N_; k+=blocksize ){
      // Load the feature from memory
      float * ff = (float*)f;
      for( int j=0; j<d_; j++ )
	for( int i=0; i<blocksize; i++ )
	  ff[ j*blocksize + i ] = k+i < N_ ? feature[ (k+i)*d_+j ] : 0.0;
			
      // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
			
      // sm contains the sum of 1..n of our faeture vector
      __m128 sm = Zero;
      for( int j=d_; j>0; j-- ){
	__m128 cf = f[j-1]*scale_factor[j-1];
	elevated[j] = sm - _mm_set1_ps(j)*cf;
	sm += cf;
      }
      elevated[0] = sm;
			
      // Find the closest 0-colored simplex through rounding
      __m128 sum = Zero;
      for( int i=0; i<=d_; i++ ){
	__m128 v = invdplus1 * elevated[i];
#ifdef __SSE4_1__
	v = _mm_round_ps( v, _MM_FROUND_TO_NEAREST_INT );
#else
	v = _mm_cvtepi32_ps( _mm_cvtps_epi32( v ) );
#endif
	rem0[i] = v*dplus1;
	sum += v;
      }
			
      // Find the simplex we are in and store it in rank (where rank describes what position coorinate i has in the sorted order of the features values)
      for( int i=0; i<=d_; i++ )
	rank[i] = Zero;
      for( int i=0; i<d_; i++ ){
	__m128 di = elevated[i] - rem0[i];
	for( int j=i+1; j<=d_; j++ ){
	  __m128 dj = elevated[j] - rem0[j];
	  __m128 c = _mm_and_ps( One, _mm_cmplt_ps( di, dj ) );
	  rank[i] += c;
	  rank[j] += One-c;
	}
      }
			
      // If the point doesn't lie on the plane (sum != 0) bring it back
      for( int i=0; i<=d_; i++ ){
	rank[i] += sum;
	__m128 add = _mm_and_ps( dplus1, _mm_cmplt_ps( rank[i], Zero ) );
	__m128 sub = _mm_and_ps( dplus1, _mm_cmpge_ps( rank[i], dplus1 ) );
	rank[i] += add-sub;
	rem0[i] += add-sub;
      }
			
      // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
      for( int i=0; i<(d_+2)*blocksize; i++ )
	barycentric[ i ] = 0;
      for( int i=0; i<=d_; i++ ){
	__m128 v = (elevated[i] - rem0[i])*invdplus1;
				
	// Didn't figure out how to SSE this
	float * fv = (float*)&v;
	float * frank = (float*)&rank[i];
	for( int j=0; j<blocksize; j++ ){
	  int p = d_-frank[j];
	  barycentric[j*(d_+2)+p  ] += fv[j];
	  barycentric[j*(d_+2)+p+1] -= fv[j];
	}
      }
			
      // The rest is not SSE'd
      for( int j=0; j<blocksize; j++ ){
	// Wrap around
	barycentric[j*(d_+2)+0]+= 1 + barycentric[j*(d_+2)+d_+1];
				
	float * frank = (float*)rank;
	float * frem0 = (float*)rem0;
	// Compute all vertices and their offset
	for( int remainder=0; remainder<=d_; remainder++ ){
	  for( int i=0; i<d_; i++ ){
	    key[i] = frem0[i*blocksize+j] + canonical[ remainder*(d_+1) + (int)frank[i*blocksize+j] ];
	  }
	  offset_[ (j+k)*(d_+1)+remainder ] = hash_table.find( key, true );
	  barycentric_[ (j+k)*(d_+1)+remainder ] = barycentric[ j*(d_+2)+remainder ];
	}
      }
    }
    _mm_free( scale_factor );
    _mm_free( f );
    _mm_free( elevated );
    _mm_free( rem0 );
    _mm_free( rank );
    delete [] barycentric;
    delete [] canonical;
    delete [] key;
		
    // Reset the SSE rounding
#ifndef __SSE4_1__
    _mm_setcsr( old_rounding );
#endif
		
    // This is normally fast enough so no SSE needed here
    // Find the Neighbors of each lattice point
		
    // Get the number of vertices in the lattice
    M_ = hash_table.size();
		
    // Create the neighborhood structure
    if(blur_neighbors_) delete[] blur_neighbors_;
    blur_neighbors_ = new Neighbors[ (d_+1)*M_ ];
		
    short * n1 = new short[d_+1];
    short * n2 = new short[d_+1];
		
    // For each of d+1 axes,
    for( int j = 0; j <= d_; j++ ){
      for( int i=0; i<M_; i++ ){
	const short * key = hash_table.getKey( i );
	for( int k=0; k<d_; k++ ){
	  n1[k] = key[k] - 1;
	  n2[k] = key[k] + 1;
	}
	n1[j] = key[j] + d_;
	n2[j] = key[j] - d_;
				
	blur_neighbors_[j*M_+i].n1 = hash_table.find( n1 );
	blur_neighbors_[j*M_+i].n2 = hash_table.find( n2 );
      }
    }
    delete[] n1;
    delete[] n2;
#else
    // Compute the lattice coordinates for each feature [there is going to be a lot of magic here
    N_ = N;
    d_ = feature_size;
    HashTable hash_table( d_, N_*(d_+1) );

    // Allocate the class memory
    if (offset_) delete [] offset_;
    offset_ = new int[ (d_+1)*N_ ];
    if (barycentric_) delete [] barycentric_;
    barycentric_ = new float[ (d_+1)*N_ ];
		
    // Allocate the local memory
    float * scale_factor = new float[d_];
    float * elevated = new float[d_+1];
    float * rem0 = new float[d_+1];
    float * barycentric = new float[d_+2];
    short * rank = new short[d_+1];
    short * canonical = new short[(d_+1)*(d_+1)];
    short * key = new short[d_+1];
		
    // Compute the canonical simplex
    for( int i=0; i<=d_; i++ ){
      for( int j=0; j<=d_-i; j++ )
	canonical[i*(d_+1)+j] = i;
      for( int j=d_-i+1; j<=d_; j++ )
	canonical[i*(d_+1)+j] = i - (d_+1);
    }
		
    // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
    float inv_std_dev = sqrtf(2.f / 3.f)*(d_+1);
    // Compute the diagonal part of E (p.5 in [Adams etal 2010])
    for( int i=0; i<d_; i++ )
      scale_factor[i] = 1.f / sqrtf( (i+2.f)*(i+1.f) ) * inv_std_dev;
		
    // Compute the simplex each feature lies in
    for( int k=0; k<N_; k++ ){
      // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
      const float * f = feature + k*feature_size;
			
      // sm contains the sum of 1..n of our faeture vector
      float sm = 0;
      for( int j=d_; j>0; j-- ){
	float cf = f[j-1]*scale_factor[j-1];
	elevated[j] = sm - j*cf;
	sm += cf;
      }
      elevated[0] = sm;
			
      // Find the closest 0-colored simplex through rounding
      float down_factor = 1.0f / (d_+1);
      float up_factor = (d_+1);
      int sum = 0;
      for( int i=0; i<=d_; i++ ){
	int rd = (int)round( down_factor * elevated[i]);
	rem0[i] = rd*up_factor;
	sum += rd;
      }
			
      // Find the simplex we are in and store it in rank (where rank describes what position coorinate i has in the sorted order of the features values)
      for( int i=0; i<=d_; i++ )
	rank[i] = 0;
      for( int i=0; i<d_; i++ ){
	double di = elevated[i] - rem0[i];
	for( int j=i+1; j<=d_; j++ )
	  if ( di < elevated[j] - rem0[j])
	    rank[i]++;
	  else
	    rank[j]++;
      }
			
      // If the point doesn't lie on the plane (sum != 0) bring it back
      for( int i=0; i<=d_; i++ ){
	rank[i] += sum;
	if ( rank[i] < 0 ){
	  rank[i] += d_+1;
	  rem0[i] += d_+1;
	}
	else if ( rank[i] > d_ ){
	  rank[i] -= d_+1;
	  rem0[i] -= d_+1;
	}
      }
			
      // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
      for( int i=0; i<=d_+1; i++ )
	barycentric[i] = 0;
      for( int i=0; i<=d_; i++ ){
	float v = (elevated[i] - rem0[i])*down_factor;
	barycentric[d_-rank[i]  ] += v;
	barycentric[d_-rank[i]+1] -= v;
      }
      // Wrap around
      barycentric[0] += 1.0f + barycentric[d_+1];
			
      // Compute all vertices and their offset
      for( int remainder=0; remainder<=d_; remainder++ ){
	for( int i=0; i<d_; i++ )
	  key[i] = rem0[i] + canonical[ remainder*(d_+1) + rank[i] ];
	offset_[ k*(d_+1)+remainder ] = hash_table.find( key, true );
	barycentric_[ k*(d_+1)+remainder ] = barycentric[ remainder ];
      }
    }
    delete [] scale_factor;
    delete [] elevated;
    delete [] rem0;
    delete [] barycentric;
    delete [] rank;
    delete [] canonical;
    delete [] key;
		
		
    // Find the Neighbors of each lattice point
		
    // Get the number of vertices in the lattice
    M_ = hash_table.size();
		
    // Create the neighborhood structure
    if(blur_neighbors_) delete[] blur_neighbors_;
    blur_neighbors_ = new Neighbors[ (d_+1)*M_ ];
		
    short * n1 = new short[d_+1];
    short * n2 = new short[d_+1];
		
    // For each of d+1 axes,
    for( int j = 0; j <= d_; j++ ){
      for( int i=0; i<M_; i++ ){
	const short * key = hash_table.getKey( i );
	for( int k=0; k<d_; k++ ){
	  n1[k] = key[k] - 1;
	  n2[k] = key[k] + 1;
	}
	n1[j] = key[j] + d_;
	n2[j] = key[j] - d_;
				
	blur_neighbors_[j*M_+i].n1 = hash_table.find( n1 );
	blur_neighbors_[j*M_+i].n2 = hash_table.find( n2 );
      }
    }
    delete[] n1;
    delete[] n2;
#endif
}

#ifdef SSE_PERMUTOHEDRAL
void Permutohedral::compute( __m128* out, const __m128* in, int value_size, int in_offset, int out_offset, int in_size, int out_size) const {
    if ( in_size == -1)  in_size = N_ -  in_offset;
    if (out_size == -1) out_size = N_ - out_offset;
		
    // Shift all values by 1 such that -1 -> 0 (used for blurring)
    __m128 * values     = (__m128*) _mm_malloc( (M_+2)*value_size*sizeof(__m128), 16 );
    __m128 * new_values = (__m128*) _mm_malloc( (M_+2)*value_size*sizeof(__m128), 16 );
		
    __m128 Zero = _mm_set1_ps( 0 );
		
    for( int i=0; i<(M_+2)*value_size; i++ )
      values[i] = new_values[i] = Zero;
		
    // Splatting
    for( int i=0;  i<in_size; i++ ){
      const __m128 * sse_val = in + i*value_size;
      for( int j=0; j<=d_; j++ ){
	int o = offset_[(in_offset+i)*(d_+1)+j]+1;
	__m128 w = _mm_set1_ps( barycentric_[(in_offset+i)*(d_+1)+j] );
	for( int k=0; k<value_size; k++ )
	  values[ o*value_size+k ] += w * sse_val[k];
      }
    }
		
    // Blurring
    __m128 half = _mm_set1_ps(0.5);
    for( int j=0; j<=d_; j++ ){
      for( int i=0; i<M_; i++ ){
	__m128 * old_val = values + (i+1)*value_size;
	__m128 * new_val = new_values + (i+1)*value_size;
				
	int n1 = blur_neighbors_[j*M_+i].n1+1;
	int n2 = blur_neighbors_[j*M_+i].n2+1;
	__m128 * n1_val = values + n1*value_size;
	__m128 * n2_val = values + n2*value_size;
	for( int k=0; k<value_size; k++ )
	  new_val[k] = old_val[k]+half*(n1_val[k] + n2_val[k]);
      }
      __m128 * tmp = values;
      values = new_values;
      new_values = tmp;
    }
    // Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
    float alpha = 1.0f / (1+powf(2, -d_));
		
    // Slicing
    for( int i=0; i<out_size; i++ ){
      __m128 * sse_val = out + i*value_size;
      for( int k=0; k<value_size; k++ )
	sse_val[ k ] = Zero;
      for( int j=0; j<=d_; j++ ){
	int o = offset_[(out_offset+i)*(d_+1)+j]+1;
	__m128 w = _mm_set1_ps( barycentric_[(out_offset+i)*(d_+1)+j] * alpha );
	for( int k=0; k<value_size; k++ )
	  sse_val[ k ] += w * values[ o*value_size+k ];
      }
    }
		
    _mm_free( values );
    _mm_free( new_values );
}
  
#endif

void Permutohedral::compute(float* out, const float* in, int value_size, int in_offset, int out_offset, int in_size, int out_size) const {
#ifdef SSE_PERMUTOHEDRAL
    if ( in_size == -1)  in_size = N_ -  in_offset;
    if (out_size == -1) out_size = N_ - out_offset;
		
    const int sse_value_size = (value_size-1)*sizeof(float) / sizeof(__m128) + 1;
    // Shift all values by 1 such that -1 -> 0 (used for blurring)
    __m128 * sse_val    = (__m128*) _mm_malloc( sse_value_size*sizeof(__m128), 16 );
    __m128 * values     = (__m128*) _mm_malloc( (M_+2)*sse_value_size*sizeof(__m128), 16 );
    __m128 * new_values = (__m128*) _mm_malloc( (M_+2)*sse_value_size*sizeof(__m128), 16 );
		
    __m128 Zero = _mm_set1_ps( 0 );
  
    for( int i=0; i<(M_+2)*sse_value_size; i++ )
      values[i] = new_values[i] = Zero;
    for( int i=0; i<sse_value_size; i++ )
      sse_val[i] = Zero;
		
    // Splatting
    for (int i = 0; i < in_size; i++) {
      memcpy(sse_val, in+i*value_size, value_size*sizeof(float));
      for( int j=0; j<=d_; j++ ){
	int o = offset_[(in_offset+i)*(d_+1)+j]+1;
	__m128 w = _mm_set1_ps( barycentric_[(in_offset+i)*(d_+1)+j] );
	for( int k=0; k<sse_value_size; k++ )
	  values[ o*sse_value_size+k ] += w * sse_val[k];
      }
    }
    // Blurring
    __m128 half = _mm_set1_ps(0.5);
    for( int j=0; j<=d_; j++ ){
      for( int i=0; i<M_; i++ ){
	__m128 * old_val = values + (i+1)*sse_value_size;
	__m128 * new_val = new_values + (i+1)*sse_value_size;
				
	int n1 = blur_neighbors_[j*M_+i].n1+1;
	int n2 = blur_neighbors_[j*M_+i].n2+1;
	__m128 * n1_val = values + n1*sse_value_size;
	__m128 * n2_val = values + n2*sse_value_size;
	for( int k=0; k<sse_value_size; k++ )
	  new_val[k] = old_val[k]+half*(n1_val[k] + n2_val[k]);
      }
      __m128 * tmp = values;
      values = new_values;
      new_values = tmp;
    }
    // Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
    float alpha = 1.0f / (1+powf(2, -d_));
		
    // Slicing
    for( int i=0; i<out_size; i++ ){
      for( int k=0; k<sse_value_size; k++ )
	sse_val[ k ] = Zero;
      for( int j=0; j<=d_; j++ ){
	int o = offset_[(out_offset+i)*(d_+1)+j]+1;
	__m128 w = _mm_set1_ps( barycentric_[(out_offset+i)*(d_+1)+j] * alpha );
	for( int k=0; k<sse_value_size; k++ )
	  sse_val[ k ] += w * values[ o*sse_value_size+k ];
      }
      memcpy( out+i*value_size, sse_val, value_size*sizeof(float) );
    }
		
    _mm_free( sse_val );
    _mm_free( values );
    _mm_free( new_values );

#else

    if ( in_size == -1)  in_size = N_ -  in_offset;
    if (out_size == -1) out_size = N_ - out_offset;
		
    // Shift all values by 1 such that -1 -> 0 (used for blurring)
    float * values = new float[ (M_+2)*value_size ];
    float * new_values = new float[ (M_+2)*value_size ];
		
    for( int i=0; i<(M_+2)*value_size; i++ )
      values[i] = new_values[i] = 0;
		
    // Splatting
    for( int i=0;  i<in_size; i++ ){
      for( int j=0; j<=d_; j++ ){
	int o = offset_[(in_offset+i)*(d_+1)+j]+1;
	float w = barycentric_[(in_offset+i)*(d_+1)+j];
	for( int k=0; k<value_size; k++ )
	  values[ o*value_size+k ] += w * in[ i*value_size+k ];
      }
    }
		
    for( int j=0; j<=d_; j++ ){
      for( int i=0; i<M_; i++ ){
	float * old_val = values + (i+1)*value_size;
	float * new_val = new_values + (i+1)*value_size;
				
	int n1 = blur_neighbors_[j*M_+i].n1+1;
	int n2 = blur_neighbors_[j*M_+i].n2+1;
	float * n1_val = values + n1*value_size;
	float * n2_val = values + n2*value_size;
	for( int k=0; k<value_size; k++ )
	  new_val[k] = old_val[k]+0.5f*(n1_val[k] + n2_val[k]);
      }
      float * tmp = values;
      values = new_values;
      new_values = tmp;
    }
    // Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
    float alpha = 1.0f / (1.f+powf(2.f, -(float)d_));
		
    // Slicing
    for( int i=0; i<out_size; i++ ){
      for( int k=0; k<value_size; k++ )
	out[i*value_size+k] = 0;
      for( int j=0; j<=d_; j++ ){
	int o = offset_[(out_offset+i)*(d_+1)+j]+1;
	float w = barycentric_[(out_offset+i)*(d_+1)+j];
	for( int k=0; k<value_size; k++ )
	  out[ i*value_size+k ] += w * values[ o*value_size+k ] * alpha;
      }
    }
		
    delete[] values;
    delete[] new_values;

#endif

}
