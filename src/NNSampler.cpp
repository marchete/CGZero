#include <immintrin.h> //SSE Extensions
#include <bits/stdc++.h> //All main STD libraries
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <atomic>
#include <random>
#include <algorithm> 
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <iostream>
#include <filesystem>
#include <regex>
//#include "fast_double_parser.h"
namespace fs = std::filesystem;
using namespace std;

const float DECAY_FACTOR = 0.02f;
const float MINIMAL_DECAY =0.65f;
float factor = 1.0f;
//#define DUMP_TXT
struct SampleInfo {
	vector<float> I;
	vector<float> P;
	float N;
	int TURN;
	int win, draw, loss;
};


#define Now() chrono::high_resolution_clock::now()
struct Stopwatch {
	chrono::high_resolution_clock::time_point c_time, c_timeout;
	void Start(int us) { c_time = Now(); c_timeout = c_time + chrono::microseconds(us); }
	void setTimeout(int us) { c_timeout = c_time + chrono::microseconds(us); }
	inline bool Timeout() {
		return Now() > c_timeout;
	}
	long long EllapsedMicroseconds() { return chrono::duration_cast<chrono::microseconds>(Now() - c_time).count(); }
	long long EllapsedMilliseconds() { return chrono::duration_cast<chrono::milliseconds>(Now() - c_time).count(); }
} stopwatch;


template <class T>
inline void hash_combine(std::size_t& seed, T const& v)
{
	seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template<typename T>
size_t hashVector(vector<T>& in) {
	size_t size = in.size();
	size_t seed = 0;
	for (size_t i = 0; i < size; i++)
		//Combine the hash of the current vector with the hashes of the previous ones
		hash_combine(seed, in[i]);
	return seed;
}
/*unordered_map<size_t, SampleInfo> samples;
vector<size_t> hashSamples;*/

unordered_map<size_t, SampleInfo> samples;
vector<pair<size_t, int>> hashSamples;

int AVAILABLE_POOL = 0;


int PROCESSED_SAMPLES = 0;
float FACTORED_SAMPLES = 0.0f;
bool processFile(fs::path file, const int INPUT_SIZE, const int OUTPUT_SIZE)
{
	auto t0 = stopwatch.EllapsedMilliseconds();
	cerr << "Processing " << file.filename();
	//Inputs + POLICY + VALUE
	ifstream F(file, std::ios::in | std::ios::binary);
	if (!F.good())
	{
		cerr << "Error reading file:" << file << endl;
		return true;
	}
	string line;
	SampleInfo S;
	S.I.resize(INPUT_SIZE);
	S.P.resize(OUTPUT_SIZE);
	int linesProcessed = 0;
	F.seekg(0);
	while (!F.eof())// (getline(F, line))
	{
		++linesProcessed;
		S.N = 1.0f;
		S.TURN=PROCESSED_SAMPLES;
		F.read(reinterpret_cast<char*>(&S.I[0]), INPUT_SIZE * sizeof(float));
		if (F.eof())
			break;
		F.read(reinterpret_cast<char*>(&S.P[0]), OUTPUT_SIZE * sizeof(float));
		F.read(reinterpret_cast<char*>(&S.N), sizeof(S.N));
		PROCESSED_SAMPLES+=(int)S.N;		
		S.N *=factor;
		FACTORED_SAMPLES+=S.N;
		for (auto& spp:S.P)
			spp*=S.N;
		S.win = 0;
		S.loss = 0;
		S.draw = 0;
		if (S.P.back() > 0.6f)
		{
			++S.win;
		}
		else if (S.P.back() < -0.6f)
		{
			++S.loss;
		}
		else ++S.draw;
		/*

				double x;
				const char * endptr = line.c_str();
				do {
					while (endptr!=nullptr && *endptr == ' ')
					{
						endptr++;
					}
					if (endptr == nullptr || *endptr == '\0')
						break;
					endptr = fast_double_parser::parse_number(endptr, &x);
					S.I.emplace_back(x);
				} while (endptr != nullptr && *endptr != '\0');


		//		istringstream iss(line);
				//copy(std::istream_iterator<double>(iss),	std::istream_iterator<double>(),	std::back_inserter(S.I));
				for (int i = (int) S.I.size()  - OUTPUT_SIZE; i < (int)S.I.size(); ++i)
				{
					S.P.push_back(S.I[i]);
				}
				if (S.P.size() != OUTPUT_SIZE)
				{
					cerr << "SIZE ERROR:" << S.P.size() << " != " << OUTPUT_SIZE << endl;
					cerr << endl;
				}
				S.I.resize(S.I.size() - OUTPUT_SIZE);
				*/
				/*size_t HASH = 0;
				for (auto&f : S.I)
				{
					hash_combine(HASH, f);
				}*/

		size_t HASH = hashVector(S.I);


		auto hasSample = samples.find(HASH);
		if (hasSample == samples.end()) //NEW
		{
			samples.emplace(HASH, S);
		}
		else {
			hasSample->second.N +=S.N;
			for (int i = 0; i < (int)hasSample->second.P.size(); ++i)
			{
				hasSample->second.P[i] += S.P[i];
			}
			hasSample->second.win += S.win;
			hasSample->second.loss += S.loss;
			hasSample->second.draw += S.draw;
		}


		--AVAILABLE_POOL;
		if (AVAILABLE_POOL == 0)
			break;

	}
	F.close();

	cerr << " Done. Lines:" << linesProcessed << " PROCESSED:" << PROCESSED_SAMPLES << " Unique:" << samples.size() << " T:" << stopwatch.EllapsedMilliseconds() - t0 <<" Decay Factor:"<<factor<< endl;
	return AVAILABLE_POOL > 0;
}

int main(int argc, char* argv[])
{
	stopwatch.Start(0);
	if (argc < 8)
	{
		cerr << "ERROR parameters: <folder> <filter> <outputFile> <TRAINING_POOL_SIZE> <TRAINING_SUBSET_SIZE> <VECTOR_INPUT_SIZE> <VECTOR_OUTPUT_SIZE> <DELETE_OLD>" << endl;
		return -1;
	}
	//string folder = argv[1];
	fs::directory_entry directory{ argv[1] };
	string filter = argv[2];
	string oFile = argv[3];
	ofstream outputfile(oFile, std::ios::out | std::ios::binary);
#ifdef DUMP_TXT
	ofstream outputF2(oFile + ".txt", std::ios::out);
#endif
	if (!outputfile.good())
	{
		cerr << "Invalid output file:" << argv[3] << endl;
		return -1;
	}

	std::regex REG(filter);
	int TRAINING_POOL_SIZE = atoi(argv[4]);
	int TRAINING_SUBSET_SIZE = atoi(argv[5]);
	int INPUT_SIZE = atoi(argv[6]);
	int OUTPUT_SIZE = atoi(argv[7]);
	int DELETEOLD = atoi(argv[8]);
	AVAILABLE_POOL = TRAINING_POOL_SIZE;

	if (TRAINING_SUBSET_SIZE > TRAINING_POOL_SIZE)
	{
		cerr << "ERROR: TRAINING_SUBSET_SIZE > TRAINING_POOL_SIZE" << endl;
		return -1;
	}

	vector<fs::path> filenames;
	for (const auto& entry : fs::directory_iterator{ directory }) {
		if (entry.is_regular_file()) {
			string sfile = entry.path().filename().string();
			if (std::regex_match(sfile, REG))
				filenames.push_back(entry.path());
		}
	}
	//PROCESS FILES
	if (filenames.size() == 0)
	{
		cerr << "ERROR: No files to process, folder:" << argv[1] << " Mask files:" << filter << endl;
		return -1;
	}
	cerr << "Processing " << filenames.size() << " files" << endl;
	sort(filenames.begin(), filenames.end(), [](const auto& lhs, const auto& rhs) {return lhs.string() > rhs.string(); });
	bool deletingFiles = false;
	for (auto& f : filenames)
	{
		if (deletingFiles)
		{
			cerr << "Removing file " << f << endl;
			remove(f);
			continue;
		}
		if (!processFile(f, INPUT_SIZE, OUTPUT_SIZE))
		{
			if (DELETEOLD == 0)
				break;
			else
				deletingFiles = true;
		}
		factor = max(MINIMAL_DECAY,factor-DECAY_FACTOR);
	}

	for (auto& M : samples) {
		hashSamples.push_back(pair<size_t, int>(M.first, M.second.N));
	}
	//PICK RANDOM SAMPLES
	if (samples.size() == 0)
	{
		cerr << "ERROR: No samples, folder:" << argv[1] << " Mask files:" << filter << endl;
		return -1;
	}
	int uniqueSamples = (int)samples.size();
	if (uniqueSamples > TRAINING_SUBSET_SIZE)
	{

		//Keep 20%best
		std::sort(hashSamples.begin(), hashSamples.end(), [](auto& a, auto& b) {return a.second > b.second; });
		ofstream st("strats.txt");
		st << "UNIQUE " << uniqueSamples << " PROCESSED:" << PROCESSED_SAMPLES << " Max Count:" << hashSamples[0].second << endl;
		for (int i = 0; i < min(200000, uniqueSamples); ++i) {
			st << hashSamples[i].second << endl;
		}
		st.close();

		int keep20 = TRAINING_SUBSET_SIZE * 20 / 100;
		int keep40 = TRAINING_SUBSET_SIZE * 40 / 100;

		int master20 = uniqueSamples * 20 / 100;
		int master40 = uniqueSamples * 40 / 100;

		std::random_device rd;
		std::mt19937 g(rd());


		/*	auto rnd1 = std::bind(std::uniform_int_distribution<int>(0,master20), g);
			for (int i = 0; i < keep20; ++i)
			{
				int j = rnd1();
				if (i != j)
				{
					swap(hashSamples[i], hashSamples[j]);
				}
			}*/
		auto rnd2 = std::bind(std::uniform_int_distribution<int>(keep20, master40), g);
		for (int i = keep20; i < keep40; ++i)
		{
			int j = rnd2();
			if (i != j)
			{
				swap(hashSamples[i], hashSamples[j]);
			}
		}
		auto rnd3 = std::bind(std::uniform_int_distribution<int>(keep40, (int)hashSamples.size() - 1), g);
		for (int i = keep40; i < TRAINING_SUBSET_SIZE; ++i)
		{
			int j = rnd3();
			if (i != j)
			{
				swap(hashSamples[i], hashSamples[j]);
			}
		}
		/*
				int master35 = uniqueSamples*35/100;
				auto rnd1 = std::bind(std::uniform_int_distribution<int>(keep20,master35), g);
				auto rnd2 = std::bind(std::uniform_int_distribution<int>(keep40,(int)hashSamples.size()-1), g);
				//Partial shuffle
				for (int i = keep20; i < keep40; ++i)
				{
					int j = rnd1();
					if (i != j)
					{
						swap(hashSamples[i], hashSamples[j]);

					}
				}
				for (int i = keep40; i < TRAINING_SUBSET_SIZE; ++i)
				{
					int j = rnd2();
					if (i != j)
					{
						swap(hashSamples[i], hashSamples[j]);
					}
				}
		*/
		hashSamples.resize(TRAINING_SUBSET_SIZE);
	}
	//SAVE TO FILE
	int totalCovered = 0;
#ifdef _MSC_VER
	std::sort(hashSamples.begin(), hashSamples.end(), [](auto& a, auto& b) {return a.second > b.second; });
#endif
	if (PROCESSED_SAMPLES < 850)
		std::sort(hashSamples.begin(), hashSamples.end(), [=](auto& a, auto& b) {return samples[a.first].TURN < samples[b.first].TURN; });
	for (auto& ttt : hashSamples)
	{
		auto& HASH = ttt.first;
		SampleInfo& s = samples[HASH];
		totalCovered += s.N;

		outputfile.write(reinterpret_cast<char*>(&s.I[0]), (int)s.I.size() * sizeof(float));
		if (s.N > 1.0f)
		{
			float divide = 1.0f / (float)s.N;
			for (auto& f : s.P) {
				f *= divide;
			}
		}
		
/*		float score = ((float)s.win + 0.5f * (float)s.draw) / (float)(s.win + s.draw + s.loss);
		float oriScore = s.P[s.P.size() - 1];
		float FACTOR_WIN= 0.1f;
		if (score > 0.53f && score < 0.96f)
		{
			score *= FACTOR_WIN;
			s.P[s.P.size() - 1] = score + (1.0f - score) * s.P.back();
		}
		else if (score < 0.47f && score > 0.04f)
		{ //Loss majority
			score = 1.0f - score;
			score *= FACTOR_WIN;
			s.P[s.P.size() - 1] = -(score + (1.0f - score) * abs(s.P.back()));
		}
		*/
		outputfile.write(reinterpret_cast<char*>(&s.P[0]), (int)s.P.size() * sizeof(float));
		outputfile.write(reinterpret_cast<char*>(&s.N), (int) sizeof(s.N));
#ifdef DUMP_TXT
		/*		if (s.N == 1)
					continue;*/

		for (auto& f : s.I) {
			outputF2 << (float)f << " ";
		}

		for (auto& f : s.P) {
			outputF2 << (float)(f) << " ";
		}

		outputF2 << " COUNT:" << s.N << " W:" << s.win << " D:" << s.draw << " L:" << s.loss <<" OriScore:"<< oriScore <<" new:"<< s.P.back()<<" TURN:"<<s.TURN<< endl;
#endif
	}
	outputfile.close();
#ifdef DUMP_TXT
	outputF2.close();
#endif
	float coverage = (float)hashSamples.size() * 100.0f / (float)uniqueSamples;
	cerr << "Ended. Processed " << PROCESSED_SAMPLES << " samples. Unique " << uniqueSamples << ". On File:" << hashSamples.size() << " Coverage:" << coverage << "% Time:" << stopwatch.EllapsedMilliseconds() << endl;
}
