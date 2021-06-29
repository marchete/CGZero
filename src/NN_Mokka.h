#pragma once
//Compile with -O3 -march=core-avx2 -mavx2 -mfma
//#define LOW_MEMORY_USE

#ifndef _MSC_VER
#pragma GCC optimize("O3","omit-frame-pointer","inline")
//#pragma GCC option("arch=native","tune=native","no-zeroupper")
#pragma GCC target("avx2,fma")
#endif
#include <immintrin.h>
#include <stdio.h>
#ifdef _WIN32
#include <malloc.h>
#endif

#include <map> //map
#include <chrono>
#include <vector>
#include <cmath>
#include <exception>
#include <string>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>
#include <ostream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>  //shared_ptr
#include <cstring> //memcpy

//#define DEBUG_MODE
#ifdef DEBUG_MODE
#include <cassert>
#define ASSERT(x) assert(x)
#else
#define ASSERT(x)
#endif

union __m256_f {
	__m256 v;
	float f[8];
};

using namespace std;

//AVX2 Exponential functions
#define MUL _mm256_mul_ps
#define FMA _mm256_fmadd_ps
#define SET _mm256_set1_ps
const __m256 exp_hi = SET(88.3762626647949f);
const __m256 exp_lo = SET(-88.3762626647949f);
const __m256 cLOG2EF = SET(1.44269504088896341f);
const __m256 cexp_C1 = SET(0.693359375f);
const __m256 cexp_C2 = SET(-2.12194440e-4f);
const __m256 cexp_p0 = SET(1.9875691500E-4f);
const __m256 cexp_p1 = SET(1.3981999507E-3f);
const __m256 cexp_p2 = SET(8.3334519073E-3f);
const __m256 cexp_p3 = SET(4.1665795894E-2f);
const __m256 cexp_p4 = SET(1.6666665459E-1f);
const __m256 cexp_p5 = SET(5.0000001201E-1f);
inline __m256 exp256_ps(const __m256& V) {
	__m256 x = V;
	__m256 tmp = _mm256_setzero_ps(), fx;
	__m256i imm0;
	__m256 one = SET(1.0f);
	x = _mm256_min_ps(x, exp_hi);
	x = _mm256_max_ps(x, exp_lo);
	fx = MUL(x, cLOG2EF);
	fx = _mm256_add_ps(fx, SET(0.5f));
	tmp = _mm256_floor_ps(fx);
	__m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
	mask = _mm256_and_ps(mask, one);
	fx = _mm256_sub_ps(tmp, mask);
	tmp = MUL(fx, cexp_C1);
	__m256 z = MUL(fx, cexp_C2);
	x = _mm256_sub_ps(x, tmp);
	x = _mm256_sub_ps(x, z);
	z = MUL(x, x);
	__m256  y = cexp_p0;
	y = FMA(y, x, cexp_p1);
	y = FMA(y, x, cexp_p2);
	y = FMA(y, x, cexp_p3);
	y = FMA(y, x, cexp_p4);
	y = FMA(y, x, cexp_p5);
	y = FMA(y, z, x);
	y = _mm256_add_ps(y, one);
	imm0 = _mm256_cvttps_epi32(fx);
	imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
	imm0 = _mm256_slli_epi32(imm0, 23);
	__m256 pow2n = _mm256_castsi256_ps(imm0);
	y = MUL(y, pow2n);
	return y;
}
//AVX2 Faster Exponential functions
inline __m256 fast_exp256_ps(const __m256& V) {
	const __m256 C1 = _mm256_set1_ps(1064872507.1541044f);
	const __m256 C2 = _mm256_set1_ps(12102203.161561485f);
	return _mm256_castsi256_ps(_mm256_cvttps_epi32(_mm256_fmadd_ps(C2, V, C1)));
}

#undef SET
#undef MUL
#undef FMA

template<typename T, std::size_t Alignment>
class aligned_allocator {
public:
	typedef T *pointer;
	typedef const T *const_pointer;
	typedef T &reference;
	typedef const T &const_reference;
	typedef T value_type;
	typedef std::size_t size_type;
	typedef ptrdiff_t difference_type;

	T *address(T &r) const {
		return &r;
	}

	const T *address(const T &s) const {
		return &s;
	}

	std::size_t max_size() const {
		return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
	}
	template<typename U>
	struct rebind {
		typedef aligned_allocator<U, Alignment> other;
	};

	bool operator!=(const aligned_allocator &other) const {
		return !(*this == other);
	}

	void construct(T *const p, const T &t) const {
		void *const pv = static_cast<void *>(p);

		new(pv) T(t);
	}

	void destroy(T *const p) const {
		p->~T();
	}

	bool operator==(const aligned_allocator &other) const {
		return true;
	}


	aligned_allocator() {}

	aligned_allocator(const aligned_allocator &) {}

	template<typename U>
	aligned_allocator(const aligned_allocator<U, Alignment> &) {}

	~aligned_allocator() {}

	T *allocate(const std::size_t n) const {
		if (n == 0) {
			return NULL;
		}
		if (n > max_size()) {
			throw std::length_error("aligned_allocator<T>::allocate() - Integer overflow.");
		}
		void *const pv = _mm_malloc(n * sizeof(T), Alignment);
		if (pv == NULL) {
			throw std::bad_alloc();
		}
		return static_cast<T *>(pv);
	}

	void deallocate(T *const p, const std::size_t n) const {
		_mm_free(p);
	}

	template<typename U>
	T *allocate(const std::size_t n, const U * /* const hint */) const {
		return allocate(n);
	}

private:
	aligned_allocator &operator=(const aligned_allocator &);
};


typedef std::vector<__m256_f, aligned_allocator<__m256_f, sizeof(__m256_f)> > aligned_vector;

class Tensor {
public:
	aligned_vector xmm;
	uint64_t xmm_size;
	uint64_t size;
	std::vector<int> shape;

	explicit Tensor(std::vector<int> shape) : shape(shape) {
		size = 1;

		for (int x : shape)
			size *= x;

		xmm_size = static_cast<uint64_t>(std::ceil(size / 8.0f));

		xmm.resize(xmm_size);

		for (int i = 0; i < xmm_size; i++) {
			xmm[i].v = _mm256_setzero_ps();
		}
	}

	explicit Tensor(aligned_vector xmm, std::vector<int> shape) : shape(shape), xmm(std::move(xmm)) {
		size = 1;

		for (int x : shape)
			size *= x;

		xmm_size = static_cast<uint64_t>(std::ceil(size / 8.0f));
	}

	Tensor() : shape({ 0 }), size(0) {}
	~Tensor() {
		xmm.clear();
		shape.clear();
	}
	/*	__m256 operator[](const uint32_t& index) const {
			return xmm[index >>3].f[index & 7];
		}*/

	void load(std::vector<float> &vec) {
		xmm.resize(xmm_size);
		xmm[xmm_size - 1].v = _mm256_setzero_ps(); //Last one to zero because it can be partially loaded
		memcpy(reinterpret_cast<char*>(xmm.data()), reinterpret_cast<char*>(vec.data()), size * sizeof(float));
	};
	explicit Tensor(std::vector<float> &vec, std::vector<int> shape) : shape(shape) {
		size = 1;

		for (int x : shape)
			size *= x;

		xmm_size = static_cast<unsigned long>(std::ceil(size / 8.0f));
		load(vec);

	};
	// Get a single element (float value) from the matrix
	inline float getElement(const uint32_t& index) const {
		return xmm[index >> 3].f[index & 7];
	}

	// Set a single element (float value) into the matrix
	// Caution: This might be an expensive operation if called multiple times. Use setChunk instead
	inline void setElement(const uint32_t& index, const float& value) {
		xmm[index >> 3].f[index & 7] = value;
	}

	// Set a whole chunk (8 float values) into the matrix
	// This is prefered over setElement
	inline void setChunk(const uint32_t& index, float *chunk) {
		ASSERT(index < xmm_size && index >= 0);
		xmm[index].v = _mm256_load_ps(chunk);
	}
	// Sets eight numbers together (__m256).
	inline void setChunk(const uint32_t& index, const __m256& chunk) {
		ASSERT(index < xmm_size && index >= 0);
		xmm[index].v = chunk;
	}

	// Retrieve a chunk from the matrix
	__m256 getChunk(const uint32_t& index) const {
		return xmm[index].v;
	}
	// Adds two matrices together, mainly for the bias.
	void add(Tensor bias, Tensor &out) {

		ASSERT(((int)size) != ((int)bias.size));
		for (int i = 0; i < bias.xmm_size; ++i) {
			out.setChunk(i, _mm256_add_ps(xmm[i].v, bias.xmm[i].v));
		}
	}
	//Subtract two matrices.
	void sub(const Tensor &a, Tensor &out) {
		ASSERT((int)size != (int)a.size);
		for (uint32_t i = 0; i < xmm_size; i++) {
			out.setChunk(i, _mm256_sub_ps(xmm[i].v, a.xmm[i].v));
		}
	}

	// Sub that takes a single value instead of an entire matrix
	void sub(const float &a, Tensor &out) {
		__m256 sub_chunk = _mm256_set1_ps(a);
		for (uint32_t i = 0; i < xmm_size; i++) {
			out.xmm[i].v = _mm256_sub_ps(xmm[i].v, sub_chunk);
		}
	}

	void mul(const float &a, Tensor &out) {
		__m256 mul_chunk = _mm256_set1_ps(a);
		for (uint32_t i = 0; i < xmm_size; i++) {
			out.xmm[i].v = _mm256_mul_ps(xmm[i].v, mul_chunk);
		}
	}

	// Calculates dot product of two matrices
	// Out is expected to be initialized with its xmm vector already resized to the correct length
	void dot_product(int kept_dim, const std::vector<float> &big_matrix_vec, uint32_t big_reserve_size,
		const Tensor &small, uint32_t chunk_range, Tensor &out) {
		uint32_t out_index = 0;
		for (uint32_t small_chunk = 0; small_chunk < small.xmm_size; small_chunk += chunk_range) {
			for (uint32_t big_chunk = 0; big_chunk < xmm_size; big_chunk += chunk_range) {
				__m256 FMA = _mm256_setzero_ps();
				for (uint32_t partial_index = 0; partial_index < chunk_range; ++partial_index) {
					FMA = _mm256_fmadd_ps(xmm[big_chunk + partial_index].v, small.xmm[small_chunk + partial_index].v, FMA);
				}
				out.setElement(out_index++, _mm_cvtss_f32(_mm256_castps256_ps128(hsums(FMA))));
			}
		}
	}
	// Prints the shape.
	std::string shape_str() const {
		std::string shape_str = "(None, ";

		for (int i = 0; i < shape.size() - 1; i++) {
			shape_str += std::to_string(shape[i]) + ", ";
		}
		shape_str += std::to_string(shape[shape.size() - 1]) + ")";
		return shape_str;
	}
	// reshapes the matrix.
	void reshape(const std::vector<int> &new_shape) {
		uint64_t new_size = 1;

		for (int x : new_shape) {
			new_size *= x;
		}
		ASSERT(size == new_size);
		shape = new_shape;
	}

	// Does horizontal sum of a chunk v
	// Only works if v is __m256, __m128 requires less instructions
	static inline __m256 hsums(__m256 const &v) {
		auto x = _mm256_permute2f128_ps(v, v, 1);
		auto y = _mm256_add_ps(v, x);
		x = _mm256_shuffle_ps(y, y, _MM_SHUFFLE(2, 3, 0, 1));
		x = _mm256_add_ps(x, y);
		y = _mm256_shuffle_ps(x, x, _MM_SHUFFLE(1, 0, 3, 2));
		return _mm256_add_ps(x, y);
	};

	// operator << : Displays contents of matrix
	friend std::ostream &operator<<(std::ostream &stream, const Tensor &matrix) {
		for (uint32_t i = 0; i < matrix.xmm_size; i++) {
			stream << std::to_string(i * 8) + " - [";
			for (uint32_t j = 0; j < 7; j++)
				stream << std::to_string(matrix.xmm[i].f[j]) + " ";
			stream << std::to_string(matrix.xmm[i].f[7]);
			stream << "]\n";
		}
		return stream;
	}



	void load(std::istream& is) {
		xmm.resize(xmm_size);
		xmm[xmm_size - 1].v = _mm256_setzero_ps(); //Last one to zero because it can be partially loaded
		is.read(reinterpret_cast<char*>(xmm.data()), size * sizeof(float));
	};
	void save(std::ostream& os) {
		os.write(reinterpret_cast<char*>(xmm.data()), size * sizeof(float));
	};

};

std::vector<float> extractValues(const std::string &file_path) {
	char c;
	float val;

	std::ifstream file;
	file.open(file_path);

	std::vector<float> values;

	// Loop until beginning of array (openining '[')
	while ((file >> c) && (c != '[')) {}

	// Keep reading values until closing ']' is met
	while ((file >> val >> c) && ((c == ',') || (c == ']'))) {
		values.push_back(val);
	}

	return values;
}

// Loads the matrices.
Tensor loadMatrix(const std::string &matrix_dir, const std::string &matrix_name) {
	std::vector<float> image_vec(extractValues(matrix_dir + "/" + matrix_name + ".ahsf"));
	std::vector<int> image_shape(3);

	std::ifstream shape_file;
	shape_file.open(matrix_dir + "/" + matrix_name + "_shape.ahsf");
	shape_file >> image_shape[0] >> image_shape[1] >> image_shape[2];

	return Tensor(image_vec, image_shape);
}

// Change the image shape to make it in columns depending on the size of the filter.

//std::function<void(const Tensor&, Tensor&)> function
void Activation_Identity(const Tensor& input, Tensor& output) {
	if (&input != &output) {
		output = input;
	}
}
void Activation_ReLU(const Tensor& input, Tensor& output) {
	__m256 zero = _mm256_setzero_ps();
	for (uint32_t i = 0; i < output.xmm_size; ++i)
	{
		output.xmm[i].v = _mm256_max_ps(input.xmm[i].v, zero);
	}
}
template<int32_t N, int32_t D> //can't pass a float as template 
void Activation_LeakyReLU(const Tensor& input, Tensor& output) {
	const __m256 zero = _mm256_setzero_ps();
	const __m256 vAlpha = _mm256_set1_ps((float)N / (float)D);
	for (uint32_t i = 0; i < output.xmm_size; ++i)
	{
		output.xmm[i].v = _mm256_add_ps(_mm256_max_ps(input.xmm[i].v, zero), _mm256_min_ps(_mm256_mul_ps(input.xmm[i].v, vAlpha), zero));
	}
}

void Activation_Softmax(const Tensor& input, Tensor& output) {
	float sum = 0.0f;
	int rem = 8-(output.size % 8);
	if (rem != 0)
	{
		for (int i = 0; i < rem; ++i)
		{
			output.setElement( (uint32_t)(output.xmm_size * 8 - i - 1),-99999999.99f);
		}
	}
	for (uint32_t i = 0; i < output.xmm_size; ++i)
	{
		output.xmm[i].v = exp256_ps(input.xmm[i].v);
#ifdef _MSC_VER
		sum += output.hsums(output.xmm[i].v).m256_f32[0];
#else
		sum += output.hsums(output.xmm[i].v)[0];
#endif
	}
	sum = 1.0f / sum;
	output.mul(sum, output);
}

void Activation_Tanh(const Tensor& input, Tensor& output) {
	for (uint32_t i = 0; i < output.size; ++i) {
		output.setElement(i, tanh(input.getElement(i)));
	}
}
void Activation_Sigmoid(const Tensor& input, Tensor& output) {
	const __m256 one = _mm256_set1_ps(1.0f);
	for (uint32_t i = 0; i < output.xmm_size; ++i)
	{
		output.xmm[i].v = exp256_ps(input.xmm[i].v);
		auto divisor = _mm256_add_ps(output.xmm[i].v, one);
		output.xmm[i].v = _mm256_div_ps(output.xmm[i].v, divisor);
	}
}
typedef std::function<void(const Tensor&, Tensor&)> Activators;
const Activators NONE = Activation_Identity;
const Activators RELU = Activation_ReLU;
const Activators TANH = Activation_Tanh;
const Activators SIGMOID = Activation_Sigmoid;
const Activators SOFTMAX = Activation_Softmax;

class Layer : public std::enable_shared_from_this<Layer> {
public:
	shared_ptr<Layer> inputLayer = nullptr;
	vector<weak_ptr<Layer>> outputLayers;

	Tensor output;
	Activators activator;
	std::string name;

	shared_ptr<Layer> link(shared_ptr<Layer> _linkLayer)
	{
		ASSERT(_linkLayer->output.size > 0);
		ASSERT(inputLayer == nullptr); //Cannot connect to multiple inputs
		inputLayer = _linkLayer;
		inputLayer->outputLayers.push_back(shared_from_this());
		//Redimension based on input Weights and Bias
		initialize(inputLayer->output.shape);
		return shared_from_this(); //return this Layer for future connections
	}
	inline shared_ptr<Layer> operator()(shared_ptr<Layer> _linkLayer)
	{
		return link(_linkLayer);
	}



	int rem;

	explicit Layer(std::string name, Activators activator = NONE) : name(std::move(name)), activator(activator) {};

	virtual ~Layer() = default;
	//	Calculate output is the function that computes the output of this layer.
	virtual void calculateOutput(Tensor &input_mat) = 0;
	virtual void predict() = 0;
	//	Precomute sets up the required matrices and variables required for calculateOutput to work.
	virtual void initialize(vector<int>& Dim) = 0;
	virtual void precompute() = 0;
	virtual void load(std::istream& is) = 0;
	virtual void save(std::ostream& os) = 0;
	virtual string getType() = 0;
	virtual int countParams() = 0;
	virtual int summary() {
		int trainableParams = countParams();
		cerr << left << setw(28) << setfill(' ') << (name + " (" + getType() + ")") << setw(26) << output.shape_str() << trainableParams << endl;
		return trainableParams;
	};
};

class ActivateLayer : public Layer {
public:
	ActivateLayer(std::string name, Activators act) : Layer(name, act) {}
	inline void calculateOutput(Tensor &input_mat) override {
		activator(input_mat, output);
	};
	inline void predict()override {
		ASSERT(inputLayer != nullptr);
		calculateOutput(inputLayer->output);
	};
	void initialize(vector<int>& Dim) override {
		output = Tensor(Dim);
	}
	void precompute()override {
		ASSERT(inputLayer != nullptr);
		output = inputLayer->output; //Same Tensor
	};
	void load(std::istream& is)override {};
	void save(std::ostream& os)override {};
	inline int countParams() override { return 0; };
};
class Softmax : public ActivateLayer {
public:
	Softmax(std::string name = "Softmax") :ActivateLayer(name, SOFTMAX) {}
	string getType() override { return "Softmax"; };
};
class Tanh : public ActivateLayer {
public:
	Tanh(std::string name = "Tanh") :ActivateLayer(name, TANH) {}
	string getType() override { return "Tanh"; };
};
class ReLU : public ActivateLayer {
public:
	ReLU(std::string name = "ReLU") :ActivateLayer(name, RELU) {}
	string getType() override { return "ReLU"; };
};
class Sigmoid : public ActivateLayer {
public:
	Sigmoid(std::string name = "Sigmoid") :ActivateLayer(name, SIGMOID) {}
	string getType() override { return "Sigmoid"; };
};
class WeightBiasLayer : public Layer { //WeightBiasLayer
protected:
	vector<Tensor> weights;
	Tensor bias;
	int num_of_outputs;
public:
	WeightBiasLayer(std::string name, Activators activator, int num_of_outputs)
		: Layer(name, activator), num_of_outputs(num_of_outputs){};

	~WeightBiasLayer() {};
	//virtual void calculateOutput(Tensor &inputMat) = 0;

	inline void predict()override {
		ASSERT(inputLayer != nullptr);
		calculateOutput(inputLayer->output);
	};

	void load(std::istream& is)override {
		for (auto&w: weights)
			w.load(is);
		bias.load(is);
	};
	void save(std::ostream& os)override {
		for (auto& w : weights)
			w.save(os);
		bias.save(os);
	};
	inline int countParams() override { return (int)(weights.size()*weights[0].size + bias.size); };
};

class Input : public Layer {
public:
	//std::vector<int> input_dim;
	Input(std::string name, std::vector<int> input_dim) : Layer(name)//, input_dim(input_dim) 
	{
		output = Tensor(input_dim);
	};
	Input(std::vector<int> input_dim) : Layer("Input")//, input_dim(input_dim) 
	{
		output = Tensor(input_dim);
	};
	~Input() {};
	void calculateOutput(Tensor &input_mat) override {
		if (&output != &input_mat)
		{
			output = input_mat;
		}
	};
	//Somebody took care to update output to the correct inputs....
	inline void predict()override {};
	void initialize(vector<int>& Dim) override {}//Already done at constructor
	void precompute()override {
		//Already done at constructor
	}
	void load(std::istream& is)override {};
	void save(std::ostream& os)override {};
	string getType() override { return "Input"; };
	inline int countParams() override { return 0; };
	int summary() override { return 0; };
	
	virtual Tensor* getInputTensor(){
		return &output;
	}
	virtual void RestartInputs() {
		for (int i = 0; i < output.xmm_size; ++i)
		{
			output.xmm[i].v = _mm256_setzero_ps();
		}
	}
	virtual void SetBit(int N) {
		output.xmm[0].f[N] = 1.0f;
	}
	virtual void UnsetBit(int N) {
		output.xmm[0].f[N] = 0.0f;
	}
	virtual void SetFloat(int N,float val) {
		output.xmm[0].f[N] = val;
	}
	virtual void UnsetFloat(int N) {
		output.xmm[0].f[N] = 0.0f;
	}
};

class Dense : public WeightBiasLayer {
public:
	Dense(std::string name, int num_of_outputs, Activators activator)
		: WeightBiasLayer(name, activator, num_of_outputs) {	};
	Dense(int num_of_outputs, Activators activator = NONE)
		: WeightBiasLayer("Dense", activator, num_of_outputs) {	};

	~Dense() {};

	void calculateOutput(Tensor &input_mat) override {
		for (int i = 0; i < output.xmm_size; ++i) {
			output.xmm[i].v = bias.xmm[i].v;
		}
		for (size_t N = 0; N < input_mat.size; N++) {
			float val = input_mat.xmm[0].f[N];
			if (val == 0.0f)
				continue;
			else if (val == 1.0f) {
				for (int i = 0; i < output.xmm_size; ++i) {
					output.xmm[i].v = _mm256_add_ps(weights[N].xmm[i].v, output.xmm[i].v);
				}
			}
			else {
				auto fm = _mm256_set1_ps(val);
				for (int i = 0; i < output.xmm_size; ++i) {
					output.xmm[i].v = _mm256_fmadd_ps(fm, weights[N].xmm[i].v, output.xmm[i].v);
				}
			}
		}
		activator(output, output);
	};

	// Sets up the Dense layer, it takes the shape of the matrix before it to compute its own matrices.
	void initialize(vector<int>& Dim) override {
		int totalSize = 1;
		for (int& n : Dim)
			totalSize *= n;
		output = Tensor(vector<int>{num_of_outputs});
		for (int i = 0; i < totalSize; ++i)
			weights.emplace_back(Tensor(vector<int>{num_of_outputs}));
		bias = Tensor(vector<int>{num_of_outputs});
#ifdef DEBUG_MODE
		cerr << "***** LAYER " << name << "******" << endl;
		//cerr << "Output " << ":" << output.shape_str()<<endl;
		cerr << "Weights " << ":" << weights.shape_str() << endl;
		cerr << "Bias " << ":" << bias.shape_str() << endl;
#endif
	}

	void precompute() override {	}

	string getType() override { return "Dense"; };
};


class Model {
public:
	vector<shared_ptr<Input>>	inputs;
	vector<shared_ptr<Layer>>	outputs;
	bool Loaded = false;
	Model(vector<shared_ptr<Input>> inputs, vector<shared_ptr<Layer>> outputs)
		: inputs(inputs)
		, outputs(outputs)
	{
		ASSERT(!inputs.empty());
		buildPath();
	}
	Model() {}
	~Model() {
		inputs.clear();
		outputs.clear();
		m_forwardPath.clear();
	};
	void predict();
	void predictSingleOutput(shared_ptr<Layer> output);
	void summary();
	void loadWeights(const std::string& f);
	void saveWeights(const std::string& f);
private:
	void buildPath();
	vector<Layer*>				m_forwardPath;  //Predicts all outputs
	map<Layer*, vector<Layer*>> m_forwardSingle; //For single output predictions
};
void Model::summary() {
	int trainableParams = 0;
	const string line = "_________________________________________________________________";
	const string line2 = "=================================================================";
	cout << line << endl;
	cout << "Layer (type)                Output Shape              Param #    " << endl;
	for (size_t i = 1; i < m_forwardPath.size(); ++i)
	{
		cout << (i == 1 ? line2 : line) << endl; //Skip 1st layer, input
		trainableParams += m_forwardPath[i]->summary();
	}
	cout << line2 << endl;
	cout << "Total params : " << trainableParams << endl;
	cout << "Trainable params : " << trainableParams << endl;
	cout << "Non - trainable params : 0" << endl;
	cout << line << endl;
}

//NOTE: prediction assumes that Input Layers are properly loaded
void Model::predict() {
	if (!Loaded)
		throw std::invalid_argument("Model weights not loaded");
	for (auto&& m : m_forwardPath) {
		m->predict();
	}
}

void Model::predictSingleOutput(shared_ptr<Layer> output) {
	if (!Loaded)
		throw std::invalid_argument("Model weights not loaded");
	for (auto&& m : m_forwardSingle[output.get()]) {
		m->predict();
	}
}

map<Layer*, vector<Layer*>> m_forwardSingle; //For single output predictions

void Model::loadWeights(const std::string& f) {
	std::ifstream is(f, std::ifstream::in | std::ios::binary);
	if (is.good())
	{
		for (auto& m : m_forwardPath) {
			m->load(is);
		}
		for (auto& m : m_forwardPath) {
			m->precompute();
		}
		Loaded = true;
	}
}
void Model::saveWeights(const std::string& f) {
	ofstream os(f, std::ifstream::out | std::ios::binary);
	for (auto&& m : m_forwardPath) {
		m->save(os);
	}
}
void Model::buildPath() {
	// build forward path by taking dependencies into account (topological sort)
	m_forwardPath.clear();
	map<Layer*, int> deps;

	for (auto l : inputs)
	{
		m_forwardPath.push_back(l.get());
	}

	for (uint32_t i = 0; i < m_forwardPath.size(); i++)
	{
		auto layer = m_forwardPath[i];
		for (auto wp : layer->outputLayers)
		{
			auto next = wp.lock().get();
			const size_t n = 1;//next->inputs.size();
			if (n > 1)
				deps[layer]++;

			if (n == 1 || n == deps[next])
				m_forwardPath.push_back(next);
		}
	}
	//Create path for single outputs. I.e. one path for policy and another for value
	m_forwardSingle.clear();
	deps.clear();
	for (auto&& O : outputs) {
		Layer* layer = O.get();
		m_forwardSingle[layer].clear();
		auto& V = m_forwardSingle[layer];
		while (layer != nullptr && deps[layer] == 0) {
			V.insert(V.begin(), layer);
			deps[layer]++;
			layer = layer->inputLayer.get();
		}
	}



}