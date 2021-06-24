#pragma GCC optimize("O3", "unroll-loops", "omit-frame-pointer", "inline")
#pragma GCC option("arch=native","tune=native","no-zeroupper")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,avx2,fma,bmi2")
#include <immintrin.h> //SSE Extensions
#include <bits/stdc++.h> //All main STD libraries
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <atomic>
#include <unordered_set>
#include "NN_Mokka.h"

using namespace std;

int VIEW_TREE_DEPTH = 2; //More depth for debug printing of MCTS Nodes.
//Use softmax for policy selfPlay
bool REPLAY_SOFTMAX_POLICY = false;

#define CFG_Game_IsSavedInMCTSNode 1
#define REMOVE_SOFTMAX_INVALID_MOVES

//#define DBG_MODE
#ifdef DBG_MODE
#include <cassert>
#define DBG(x) x
#define ASSERT(x) assert(x)
bool printSelect = false;
#else
#define DBG(x)
#define ASSERT(x)
#endif

//Alphazero Hyperparameters. 
enum mcts_mode { selfplay, pit, submit };
struct MCTS_Conf {
	//All these initial values will be replaced with the parameters from 
	float cpuct_base = 1.0f;
	float cpuct_inc = 0.0f;
	float cpuct_limit = 1.0f;

	float dirichlet_noise_epsilon = 0.20f;
	float dirichlet_noise_alpha = 1.0f;	// Dirichlet alpha = 10 / n --> Max expected moves
	float dirichlet_decay = 0.00f;
	int num_iters_per_turn = 800;
	float simpleRandomRange = 0.00f;
	bool useTimer = false;
	bool useHeuristicNN = false;
	float PROPAGATE_BASE = 0.7f; //Propagate "WIN/LOSS" with 70% at start
	float PROPAGATE_INC = 0.3f; //Linearly increase +30% until endgame
	int POLICY_BACKP_FIRST = 10; //Similarly , but with percentage of turns, first 30% of turns doesn't have any "temperature",
	int POLICY_BACKP_LAST = 10; //from 30% to (100-10=90%) I linearly sharpen policy to get only the best move, 


	mcts_mode mode;
	MCTS_Conf() {}
	MCTS_Conf(float _cpuct_base, float _cpuct_inc, float _cpuct_limit, float _dirichlet_noise_epsilon,
		float _dirichlet_noise_alpha, float _dirichlet_decay, bool _useTimer, int _num_iters_per_turn,
		float _simpleRandomRange, float _PROPAGATE_BASE, float _PROPAGATE_INC, int _POLICY_BACKP_FIRST, int _POLICY_BACKP_LAST,
		mcts_mode _mode) {
		cpuct_base = _cpuct_base;
		cpuct_inc = _cpuct_inc;
		cpuct_limit = _cpuct_limit;
		dirichlet_noise_epsilon = _dirichlet_noise_epsilon;
		dirichlet_noise_alpha = _dirichlet_noise_alpha;
		dirichlet_decay = _dirichlet_decay;
		useTimer = _useTimer;
		num_iters_per_turn = _num_iters_per_turn;
		mode = _mode;
		simpleRandomRange = _simpleRandomRange;
		PROPAGATE_BASE = _PROPAGATE_BASE;
		PROPAGATE_INC = _PROPAGATE_INC;
		POLICY_BACKP_FIRST = _POLICY_BACKP_FIRST;
		POLICY_BACKP_LAST = _POLICY_BACKP_LAST;
	}

	string print() {
		string otp = "Conf:";
		otp += " cpuct_base:" + to_string(cpuct_base);
		otp += " CPUCT_inc:" + to_string(cpuct_inc);
		otp += " cpuct_limit:" + to_string(cpuct_limit);
		otp += " DN_e:" + to_string(dirichlet_noise_epsilon);
		otp += " DN_A:" + to_string(dirichlet_noise_alpha);
		otp += " DN_d:" + to_string(dirichlet_decay);
		otp += " Iters:" + to_string(num_iters_per_turn);
		otp += " Mode:" + to_string(mode);
		otp += " Rnd:" + to_string(simpleRandomRange);
		otp += " PROPAGATE_BASE:" + to_string(PROPAGATE_BASE);
		otp += " PROPAGATE_INC:" + to_string(PROPAGATE_INC);
		otp += " POLICY_BACKP_FIRST:" + to_string(POLICY_BACKP_FIRST);
		otp += " POLICY_BACKP_LAST:" + to_string(POLICY_BACKP_LAST);
		return otp;

	}
} default_conf;


/* From leela chess zero. cpuct is not a constant at all.....
  float GetCpuct(bool at_root) const { return at_root ? kCpuctAtRoot : kCpuct; }
  float GetCpuctBase(bool at_root) const {	return at_root ? kCpuctBaseAtRoot : kCpuctBase;  }
  float GetCpuctFactor(bool at_root) const {	return at_root ? kCpuctFactorAtRoot : kCpuctFactor;}
inline float ComputeCpuct(const SearchParams& params, uint32_t N,
	bool is_root_node) {
	const float init = params.GetCpuct(is_root_node);
	const float k = params.GetCpuctFactor(is_root_node);
	const float base = params.GetCpuctBase(is_root_node);
	return init + (k ? k * FastLog((N + base) / base) : 0.0f);
}
*/
//Initial configs, selfplay and pit are irrelevant because they are override by command line arguments.
MCTS_Conf selfPlay_Mode(1.0f, 0.0f, 1.0f, 0.25f, 1.0f, 0.01f, false, 1200, 0.00f, 0.7f, 0.3f, 10, 10, mcts_mode::selfplay);
MCTS_Conf pit_Mode(1.0f, 0.0f, 1.0f, 0.05f, 1.0f, 0.007f, false, 1200, 0.00f, 0.7f, 0.3f, 10, 10, mcts_mode::pit);
//***********NOTE: *****************/
//SUBMIT MODE CONFIG IS IMPORTANT!!! In Codingame you can pass parameters too but by default it will pick these values.
MCTS_Conf submit_Mode(2.00f, 0.0f, 2.00f, 0.00f, 1.0f, 0.0000f, true, 100, 0.02f, 0.0f, 0.0f, 0, 0, mcts_mode::submit);


#ifdef _MSC_VER
#define ALIGN __declspec(align(32))
#else 
#define ALIGN __attribute__ ((aligned(32)))
#endif

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



//{ Fast Random
class Random {
public:
	const uint64_t K_m = 0x9b60933458e17d7d;
	const uint64_t K_a = 0xd737232eeccdf7ed;

	uint64_t seed;
	Random(uint64_t SEED) {
		seed = SEED;
	}

	Random() {
		random_device rd;
		mt19937 e2(rd());
		uniform_int_distribution<int32_t> dist(numeric_limits<int32_t>::min(), numeric_limits<int32_t>::max());
		seed = dist(e2);
		seed = (seed << 32) + dist(e2);
	}
	inline uint32_t xrandom() {
		//PCG 
		seed = seed * K_m + K_a;
		return (uint32_t)(seed >> (29 - (seed >> 61)));
	}

	inline uint32_t NextInt(const uint32_t& range) {
		seed = seed * K_m + K_a;
		uint64_t random32bit = (seed >> (29 - (seed >> 61))) & 0xFFFFFFFF;
		random32bit *= range;
		return	(uint32_t)(random32bit >> 32);
	}
	inline int32_t NextInt(const int32_t& a, const int32_t& b) {
		return	(int32_t)NextInt((uint32_t)(b - a + 1)) + a;
	}
	inline float NextFloat() {
		uint32_t xr = xrandom();
		if (xr == 0U) return 0.0f;
		union
		{
			float f;
			uint32_t i;
		} pun = { (float)xr };
		pun.i -= 0x10000000U;
		return	pun.f;
	}
	inline float NextFloat(const float& a, const float& b) {
		return NextFloat() * (b - a) + a;
	}

};

/****************************************  GLOBAL VARIABLES **************************************************/
int SIMCOUNT = 0;
int THREADS = 0;
const int MAX_TURNS = 199;
const int ROLLOUT_DEPTH = 0;

/***********************************NN Predict Caching****************************/
struct CachedNNEval {
	Tensor output;
	float nVal;
};
bool USE_NNEVAL_CACHE = false;
mutex mutex_NNCache;
atomic<uint64_t> NNCACHE_TOTAL;
atomic<uint64_t> NNCACHE_MISS;
atomic<uint64_t> NNCACHE_HIT;
unordered_map<uint64_t, CachedNNEval> cacheNNEval;
/****************************************  VIRTUAL GAME AND MOVE **************************************************/

//C# like structures _CLASS are the real objects, CLASS are pointer to these objects
class _Game;
typedef shared_ptr<_Game> Game;

typedef uint8_t Move;
const Move INVALID_MOVE = 254;

static string PrintMove(uint8_t m, const Game& S) {
	return to_string((int)m);
}

static int CG_ID = 0;
static const int POLICY_SIZE = 6;

class _Game : public std::enable_shared_from_this<_Game> {
public:
	/********************************* GENERIC METHODS (INTERFACE). ALWAYS IMPLEMENT THEM  **********************************/
//Creating and cloning
	_Game();
	_Game(const Game& original);
	_Game(const _Game& original);
	~_Game() {}
	void CopyFrom(const Game& original);
	void CopyFrom(const _Game& original);
	Game Clone();
	void Reset();

	Game getptr() { return shared_from_this(); }
	//Reading data
	void readConfig(Stopwatch& t);
	void readTurn(Stopwatch& t);
	int getPackSize()noexcept;
	void Pack(uint8_t* g)noexcept;
	void Unpack(uint8_t* g)noexcept;
	//
	string Print();
	float getInitialTemperature() {
		if (turn < 30 || turn == 255)  //More randomness the first 5 turns
			return 1.0f;
		else return 0.0f;
	}

	bool isEndGame();
	int getWinnerID();
	int getIDToPlay();
	int getTimeLimit();
	int getPlayerCount();
	void swapPlayers();
	uint64_t CalcHash(const int& playerID)noexcept;

	void getPossibleMoves(const int& concurrentUnitID, vector<Move>& Possible_Moves, const int& _depth);
	void Simulate(const vector<Move>& concurrentUnitMoves);
	void Simulate(const Move& singleMove);

	bool Equals(const Game& g);
	bool Equals(const _Game& g);
	//Evaluation
	float EvalPlayer(const int& playerID, const int& _depth);
	//Unused, left only for testing purposes
	float EvalHeuristic(const int& playerID, const int& _depth);


#ifdef REMOVE_SOFTMAX_INVALID_MOVES
	static Model CreateNNModel(bool activeSoftMax = false);
#else
	static Model CreateNNModel(bool activeSoftMax = true);
#endif
	static int getInputDimensions();
	void setNNInputs(Model& model, const int& playerID);
	float EvalNN(Model& model, const int& playerID, const int& _depth);
	void predict(Model& model, const int& playerID, Tensor** policy, float& nVal);

	/********************************* GAME SPECIFIC METHODS AND VARIABLES **********************************/

	/*   should be removed when implementing a new game*/
	static const int PACKED_SIZE = 16;
	static const int PLAYERS = 2;
	static const int COLUMNS = 6;


	static int getPolicySize() {
		return COLUMNS;
	}

	union {
#ifndef _MSC_VER
		__m128i v;
#endif
		uint64_t WW[2]; //For simpler packing and unpacking
		struct {
			//LL[0]
			uint8_t turn;
			uint8_t score0;
			uint8_t cell0[6];
			//LL[1]
			uint8_t gameEnded : 1;
			uint8_t idToPlay : 1;
			uint8_t lastMove : 3;
			uint8_t swapped : 1; //Sometimes I swap players, so I track the swap here.
			uint8_t freeBits : 2;//2 bits available
			uint8_t score1;
			uint8_t cell1[6];
		};
	};

};

const uint8_t IS_LEAF = 0; //Simpler to reset
//In restrospect that was a terrible idea. I should have a different variable for node state
const uint8_t NO_CHILDREN = 255;


inline int _Game::getPackSize()noexcept { return _Game::PACKED_SIZE; }
inline int _Game::getIDToPlay() { return idToPlay; }

inline void _Game::CopyFrom(const Game& original) {
	WW[0] = original->WW[0];
	WW[1] = original->WW[1];
}
inline void _Game::CopyFrom(const _Game& original) {
	WW[0] = original.WW[0];
	WW[1] = original.WW[1];
}


int _Game::getTimeLimit() {
	if (turn == 0) return 910 * 1000; else  return 44 * 1000;
}

void _Game::swapPlayers() {
	for (int i = 0; i < COLUMNS; ++i)
	{
		swap(cell0[i], cell1[i]);
	}
	swap(score0, score1);
	idToPlay = 1 - idToPlay;
	swapped = 1 - swapped;
}


int  _Game::getPlayerCount() { return PLAYERS; }

void  _Game::Reset() {
	//Fill 0's
	WW[0] = 0ULL; WW[1] = 0ULL;
	turn = -1;
	for (int i = 0; i < COLUMNS; ++i) {
		cell0[i] = 4;
		cell1[i] = 4;
	}
	freeBits = 0;
	CalcHash(0);
	turn = 0;
}
void  _Game::readConfig(Stopwatch& t) {
	Reset();
	turn = 255;
}
void  _Game::readTurn(Stopwatch& t) {
	++turn;
	idToPlay = 0;
	int ingameSeeds = 0;
	for (int P = 0; P < 2; ++P)
	{
		for (int i = 0; i < 6; ++i)
		{
			int seed;
			cin >> seed; cin.ignore();
			if (i == 0 && P == 0)
			{
				t.Start(getTimeLimit());
			}
			ingameSeeds += seed;

			if (turn == 0 && seed != 4)
			{
				CG_ID = 1;
			}

			if (P == 0)
			{
				cell0[i] = seed;
			}
			else
			{
				cell1[i] = seed;
			}

		}
	}
	if (turn == 0)
	{
		if (CG_ID > 0)
		{
			++turn;
		}
		cerr << "I'm Player" << (CG_ID > 0 ? "2" : "1") << endl;
	}
	else {
		//Recover enemy score from the game state
		int Calculated = 48 - ingameSeeds - score0;
		if (Calculated != score1)
		{
			if (Calculated >= 0)  score1 = Calculated;
		}
	}
	CalcHash(0);
	cerr << "Turn:" << (int)turn << " Points:" << (int)score0 << " " << (int)score1 << endl;
	cerr << Print() << endl;
}


string  _Game::Print() {
	string s = "";
	for (int i = 0; i < 6; ++i)
	{
		s += to_string(cell1[5 - i]) + " ";
	}
	s += " SC:" + to_string(score1) + "\n";
	for (int i = 0; i < 6; ++i)
	{
		s += to_string(cell0[i]) + " ";
	}
	s += " SC:" + to_string(score0);
	s += " idToPlay:" + to_string(idToPlay) + " Frame:" + to_string(turn + 2) + " Turn:" + to_string(turn) + " EndGame:" + to_string(gameEnded);
	return s;
}

void  _Game::getPossibleMoves(const int& concurrentUnitID, vector<Move>& Possible_Moves, const int& _depth)
{
	Possible_Moves.resize(0);
	auto CH = (WW[1 - idToPlay] >> 16);
	//Check if any enemy cell is !=0, removing turn and score bytes
	bool opponentCanPlay = (CH != 0ULL);

	uint8_t* pCell = (idToPlay == 0 ? &cell0[0] : &cell1[0]);
	if (opponentCanPlay)
		for (int i = 0; i < 6; i++) {
			if (pCell[i] > 0)
				Possible_Moves.push_back((Move)i);
		}
	else
		for (int i = 0; i < 6; i++) {
			if (pCell[i] >= 6 - i)
				Possible_Moves.push_back((Move)i);
		}

	ASSERT((int)Possible_Moves.size() <= 6);

	//Special case. No moves leads to endgame, and the player captures stones. Weird rule
	if (Possible_Moves.size() == 0)
	{
		if (!gameEnded)//Add scores when not
		{
			uint8_t sumS = 0;
			for (int i = 0; i < 6; ++i)
			{
				sumS += cell0[i] + cell1[i];
				cell0[i] = 0;
				cell1[i] = 0;
			}
			if (idToPlay == 0)
				score0 += sumS;
			else score1 += sumS;
			gameEnded = true;
		}
	}

}

inline bool  _Game::isEndGame() {
	if (gameEnded)
		return true;
	if (turn >= MAX_TURNS && turn < 250)
	{
		gameEnded = true;
		return true;
	}
	//Winning conditions

	if (score1 > 24) {
		gameEnded = true;
		return true;
	}
	else if (score0 > 24) {
		gameEnded = true;
		return true;
	}

	return false;
}

inline int  _Game::getWinnerID() {
	if (!gameEnded)
		return 2;
	if (score1 > score0) {
		return 1;
	}
	else if (score0 > score1) {
		return 0;
	}
	return 2;
}

void  _Game::Simulate(const vector<Move>& concurrentUnitMoves) {
	Simulate(concurrentUnitMoves[0]);
}
void  _Game::Simulate(const Move& singleMove) {
	//I hate the game rules. A lot of yes but no.
	//Grand Slam and forfeits are ugly.
	lastMove = singleMove;
	int toMove = (int)lastMove;

	uint8_t* myCells;
	uint8_t& myScore = (idToPlay == 0 ? score0 : score1);
	uint8_t* enemyCells;
	if (idToPlay == 0)
	{
		myCells = &cell0[0];
		enemyCells = &cell1[0];
	}
	else {
		myCells = &cell1[0];
		enemyCells = &cell0[0];
	}

	bool canCapture = false;
	int sow_pos = toMove;
	{ //sowing
		int	seeds_to_place = myCells[toMove];
		int fullSow = 0;
		if (seeds_to_place > 11) //kroo
		{
			int fullSow = (seeds_to_place - 1) / 11;
			for (int i = 0; i < 6; ++i) //Fast sow
			{
				cell0[i] += fullSow;
				cell1[i] += fullSow;
			}
			seeds_to_place = seeds_to_place % 11;
			if (seeds_to_place == 0)
				seeds_to_place = 11;
		}
		uint8_t* sowCell = myCells;
		uint8_t* swapCell = enemyCells;
		while (seeds_to_place-- > 0)
		{
			++sow_pos;
			if (sow_pos >= COLUMNS) {
				sow_pos = 0;
				swap(sowCell, swapCell);
			}
			++sowCell[sow_pos];
		}
		canCapture = (sowCell == enemyCells);
	}
	myCells[toMove] = 0;

	if (canCapture && (enemyCells[sow_pos] == 2 || enemyCells[sow_pos] == 3))
	{
		const uint64_t mask_4 = 0xFCFCFCFCFCFCUL;
		uint64_t a = WW[1 - idToPlay] >> 16; //Only enemy cells
		bool canCapture = true;

		bool untouchedHasSeeds = (a >> (8 * (1 + sow_pos))) != 0ULL;
		bool touchedBiggerThan3 = ((a & mask_4) != 0ULL);

		if (!untouchedHasSeeds && !touchedBiggerThan3)
		{
			int total = 0;
			int capture = 0;

			for (int i = COLUMNS - 1; i >= 0; --i)
			{
				total += enemyCells[i];

				if (i <= sow_pos)
				{
					if (enemyCells[i] != 2 && enemyCells[i] != 3)
						break;
					capture += enemyCells[i];
				}
				if (total != capture)
					break;
			}
			canCapture = (total != capture);
		}
		if (canCapture)
		{
			for (int i = sow_pos; i >= 0; --i)
			{
				if (enemyCells[i] != 2 && enemyCells[i] != 3)
					break;
				myScore += enemyCells[i];
				enemyCells[i] = 0;
			}
		}
	}
	idToPlay = 1 - idToPlay;
	++turn;
	++SIMCOUNT;

	return;
}


inline uint64_t splittable64(uint64_t x)
{
	x ^= x >> 30;
	x *= UINT64_C(0xbf58476d1ce4e5b9);
	x ^= x >> 27;
	x *= UINT64_C(0x94d049bb133111eb);
	x ^= x >> 31;
	return x;
}

uint64_t _Game::CalcHash(const int& playerID)noexcept {
	//Generic purpose hashing of 128bits
	uint64_t C0 = WW[0] & 0xFFFFFFFFFFFFFF00ULL;
	C0 |= (uint64_t)playerID;
	uint64_t lower_hash = splittable64(C0); //Remove turn, add player id
	uint64_t upper_hash = splittable64(WW[1]/* & 0xFFFFFFFFFFFFFFFFULL*/); //Remove extra info?
	uint64_t rotated_upper = upper_hash << 31 | upper_hash >> 33;
	return lower_hash ^ rotated_upper;
	/* A best way should be to pack all useful bits. Up to 12 bits per cell, up to score 31, then some info about the biggest seedcount.
	uint64_t Hash = 0ULL;
	const uint64_t MASK_CELLA = 0x0E010E010E010000ULL; //First bit of 6 cells
	const uint64_t MASK_CELLB = 0x010E010E010E0000ULL; //remaining bits 3*6 = 18

	Hash = _pext_u64(WW[0], MASK_CELLA)  //0  - 12 bits from cell0
		+ (_pext_u64(WW[1], MASK_CELLA) << 12) // 12 bits from cell1
		+ (_pext_u64(WW[0], MASK_CELLB) << 24) // 12 bits from cell0
		+ (_pext_u64(WW[1], MASK_CELLB) << 36) // 12 bits from cell1
		;
	auto A = _pext_u64(WW[0], MASK_CELLA); //0  - 12 bits from cell0
	auto B = (_pext_u64(WW[1], MASK_CELLA) << 12); // 12 bits from cell1
	auto C = (_pext_u64(WW[0], MASK_CELLB) << 24); // 12 bits from cell0
	auto D = (_pext_u64(WW[1], MASK_CELLB) << 36); // 12 bits from cell1
	
#ifdef _MSC_VER
	cerr << bitset<64>(A)<<" : A" << endl;
	cerr << bitset<64>(B) << " : B" << endl;
	cerr << bitset<64>(C) << " : C" << endl;
	cerr << bitset<64>(D) << " : D" << endl;
	cerr << bitset<64>(Hash) << " : Hash MID:" <<Hash<< endl;
#endif

	//Only 1 score is saved
	if (score1 > score0)
	{
		Hash += (1ULL << 48) + (((uint64_t)(score1 > 31 ? 31 : score1)) << 49);
	}
	else {
		Hash += (((uint64_t)(score0 > 31 ? 31 : score0)) << 49);
	}
	//cerr << bitset<64>(Hash) << endl;
	uint64_t BiggestIndex = 0ULL;
	uint8_t BiggestScore = cell0[0];
	for (uint64_t i = 0; i < 6; ++i)
	{
		if (cell0[i] > BiggestScore)
		{
			BiggestIndex = i;
			BiggestScore = cell0[i];
		}
		if (cell1[i] > BiggestScore)
		{
			BiggestIndex = 6 + i;
			BiggestScore = cell1[i];
		}
	}
	Hash += (((uint64_t)BiggestIndex) << 54); //4 bits
	//cerr << bitset<64>(Hash) << endl;
	Hash += (((uint64_t)(BiggestScore >> 4)) << 58); //1 bit for higher
	//cerr << bitset<64>(Hash) << endl;
	uint64_t BiggestIndex2 = 0ULL;
	BiggestScore = 0;
	for (uint64_t i = 0; i < 6; ++i)
	{
		if (i != BiggestIndex && cell0[i] > BiggestScore)
		{
			BiggestIndex2 = i;
			BiggestScore = cell0[i];
		}
		if (i != BiggestIndex + 6 && cell1[i] > BiggestScore)
		{
			BiggestIndex2 = 6 + i;
			BiggestScore = cell1[i];
		}
	}
	Hash += (((uint64_t)BiggestIndex2) << 59); //4 bits
#ifdef _MSC_VER
	cerr << bitset<64>(Hash) << " : Hash FIN:" << Hash << endl;
#endif
	return Hash;
	*/
}

//Save gamestate on a variable on MCTS Node
void _Game::Pack(uint8_t* g)noexcept {
	uint64_t* target = (uint64_t*)g;
	target[0] = WW[0];
	target[1] = WW[1];
}
//Restore from MCTS Node.
void _Game::Unpack(uint8_t* g)noexcept {
	uint64_t* src = (uint64_t*)g;
	WW[0] = src[0];
	WW[1] = src[1];
}

_Game::_Game() {
}
_Game::_Game(const Game& original) {
	CopyFrom(original);
}
_Game::_Game(const _Game& original) {
	CopyFrom(original);
}
//Each cell is coded as [0-24], and score as [0-26]. If score goes >26, it's saved as 26 in the inputs
int  _Game::getInputDimensions() { return 6 * 2 * 24 + 2 * 27; }

//TODO: Here is the Neural Network Model, it must match exactly the model in tensorflow.
// That's a critical part, it must be the same trainable parameters count AND in the same order.
// Ensure that predictions from Tensorflow and C++ are the same.
Model _Game::CreateNNModel(bool activeSoftMax) {

	shared_ptr<Input> input;
	shared_ptr<Layer> value, policy;
	shared_ptr<Layer> x, v1, p1;
#define NN(M_M) make_shared<M_M>
	input = NN(Input)(vector<int>{getInputDimensions()});
	x = (*NN(Dense)("Dense1", TODOTODOTODO, RELU))(input);
//	x = (*NN(Dense)("Dense2", TODOTODOTODO, RELU))(x);
	v1 = (*NN(Dense)("v1", TODOTODOTODO, RELU))(x);
//	v1 = (*NN(Dense)("v2", TODOTODOTODO, RELU))(v1);	
	p1 = (*NN(Dense)("p1", TODOTODOTODO, RELU))(x);
//	p1 = (*NN(Dense)("p2", TODOTODOTODO, RELU))(p1);	
	value = (*NN(Dense)("Value", 1, TANH))(v1);
	policy = (*NN(Dense)("Policy", POLICY_SIZE, (activeSoftMax ? SOFTMAX : NONE)))(p1); //it should be softmax, but we are normalizing after move restrictions
#undef NN
	Model model({ input }, { value ,policy });
	return model;
}

Game _Game::Clone() {
	Game g = std::make_shared<_Game>();
	g->CopyFrom(*this);
	return g;
}
inline bool _Game::Equals(const Game& g)
{
	return Equals(*g);
}

inline bool _Game::Equals(const _Game& g) {
	bool resultado = (WW[0] == g.WW[0]) && ((WW[1] >> 8) == (g.WW[1] >> 8));
	return resultado;
}

//*************************************************** SAMPLE REPLAY BUFFERS **************************************************************************//

//SELF-PLAY: SAMPLES
struct ReplayMove {
	bool ignoreDontSave;
	vector<float> gamestate; //Tensor gamestate;
	vector<float> policy; //Tensor policy;
	float meanScore;
	int selectedMove;
	int validMovesNr;


	vector<float> originalpolicy; //Tensor policy;
	float factorBPP;
	float originalValue;
	float backtrackValue;

	int currIDToPlay;
	int tmpIDToPlay;
	int tmpSwapped;
	_Game testGame;
	string SearchResult = "";
};
struct SampleInfo {
	vector<float> I;
	vector<float> P;
	int N;
	int win, draw, loss;
};

struct SamplesFile {
	string file;
	unordered_map<size_t, SampleInfo> samples;
	SamplesFile(string _file) {
		file = _file;
	}
};

vector<SamplesFile> samplesPerFile;
std::mutex mutex_selfGames;

SamplesFile* getSampleFile(string file) {
	SamplesFile* sFile = nullptr;
	for (auto&s : samplesPerFile)
	{
		if (s.file == file)
		{
			sFile = &s;
			break;
		}
	}
	if (sFile == nullptr)
	{
		SamplesFile newFile(file);
		samplesPerFile.emplace_back(newFile);
		sFile = &samplesPerFile.back();
	}
	return sFile;
}

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
void insertNewSample(SamplesFile* sFile, SampleInfo& S) {
	size_t HASH = hashVector(S.I);

	auto hasSample = sFile->samples.find(HASH);
	if (hasSample == sFile->samples.end()) //NEW
	{
		sFile->samples.emplace(HASH, S);
	}
	else {
		hasSample->second.N += S.N;
		for (int i = 0; i < (int)hasSample->second.P.size(); ++i)
		{
			hasSample->second.P[i] += S.P[i];
		}
		hasSample->second.win += S.win;
		hasSample->second.loss += S.loss;
		hasSample->second.draw += S.draw;
	}
}

int processSamplesFile(string file, const int INPUT_SIZE, const int OUTPUT_SIZE)
{
	auto t0 = stopwatch.EllapsedMilliseconds();
	cerr << "Processing " << file;
	//Inputs + POLICY + VALUE
	mutex_selfGames.lock();
	SamplesFile* sFile = getSampleFile(file);
	ifstream F(file, std::ios::in | std::ios::binary);
	if (!F.good())
	{
		mutex_selfGames.unlock();
		cerr << "Error reading file:" << file << endl;
		return true;
	}
	//Create space


	string line;
	SampleInfo S;
	S.I.resize(INPUT_SIZE);
	S.P.resize(OUTPUT_SIZE);
	int linesProcessed = 0;
	F.seekg(0);
	int PROCESSED_SAMPLES = 0;


	while (!F.eof())// (getline(F, line))
	{
		++PROCESSED_SAMPLES;
		++linesProcessed;
		S.N = 1;
		F.read(reinterpret_cast<char*>(&S.I[0]), INPUT_SIZE * sizeof(float));
		if (F.eof())
			break;
		F.read(reinterpret_cast<char*>(&S.P[0]), OUTPUT_SIZE * sizeof(float));
		float fN = (float)S.N;
		F.read(reinterpret_cast<char*>(&fN), sizeof(fN));
		S.win = 0;
		S.loss = 0;
		S.draw = 0;
		if (S.P.back() > 0.45f)
		{
			++S.win;
		}
		else if (S.P.back() < -0.45f)
		{
			++S.loss;
		}
		else ++S.draw;
		insertNewSample(sFile, S);
	}
	mutex_selfGames.unlock();
	F.close();

	cerr << " Done. Lines:" << linesProcessed << " PROCESSED:" << PROCESSED_SAMPLES << " Unique:" << sFile->samples.size() << " T:" << stopwatch.EllapsedMilliseconds() - t0 << endl;
	return true;
}

bool saveSamplesFile(string file) {
	SamplesFile* sFile = getSampleFile(file);
	if (sFile == nullptr)
	{
		cerr << "Error, samples for  " << file << " not found!" << endl;
		return false;
	}
	ofstream outputfile(file, std::ios::out | std::ios::binary);
	if (!outputfile.good())
	{
		cerr << "Invalid output file:" << file << endl;
		return false;
	}

	for (auto& ttt : sFile->samples)
	{
		SampleInfo& s = ttt.second;
		outputfile.write(reinterpret_cast<char*>(&s.I[0]), (int)s.I.size() * sizeof(float));
		//Averaging
		if (s.N > 1)
		{
			float divide = 1.0f / (float)s.N;
			for (auto& f : s.P) {
				f *= divide;
			}
		}
		outputfile.write(reinterpret_cast<char*>(&s.P[0]), (int)s.P.size() * sizeof(float));
		float fN = (float)s.N;
		outputfile.write(reinterpret_cast<char*>(&fN), (int) sizeof(fN));
	}
	outputfile.close();
	return true;
}

struct ReplayGame {
	vector<ReplayMove> moves;
	float reward; // -1.0 to 1.0
	_Game game;
};
struct ReplayBuffer {
	vector<ReplayGame> games;
};


//*************************************************** MCTS Nodes**************************************************************************//
Random SMIT_rnd = Random();
struct MCTS_Node;
struct PRECACHE_PACK;
struct PRECACHE;

//Fast functions for MCTS calculation of UCT
inline float fastlogf(const float& x) {
	union { float f; uint32_t i; } vx = { x };
#pragma warning( push )
#pragma warning( disable : 4244 )
	float y = vx.i;
#pragma warning( pop )
	y *= 8.2629582881927490e-8f;
	return(y - 87.989971088f);
}
inline float fastsqrtf(const float& x) {
	union { int i; float x; }u;
	u.x = x;
	u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
	return(u.x);
}
inline float sqrt_log(const int& n) {
	return fastsqrtf(fastlogf((float)n));
}
inline float inv_sqrt(const int& x) { return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss((float)x))); }
inline float inv_sqrt(float x) { return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))); }

inline float fastinv(const int& x) { return _mm_cvtss_f32(_mm_rcp_ps(_mm_set_ss((float)x))); }


const uint16_t PRECACHE_NODECOUNT_PACK = 16384;// 65535; NN doesn't need that much nodes
struct NodeIndex {
	uint16_t BlockID; //This allows tree reuse without exhausting nodes or collisions
	uint16_t FirstChild;
};

const uint8_t Status_GAMESTATE_SAVED = 1 << 0;

//Node used in MCTS, it holds the classic info + NN value + saved gamestate
struct MCTS_Node {
	float sumScore;
	float maxScore;
	float nnValue;
	float policy;
	int visits;
	Move action; //To recreate gamestate
	NodeIndex ChildIndex;
	uint8_t ChildCount; //8 - We shouldn't have more than 253 children.
	MCTS_Node* parent;
	union {
		uint8_t Status;
		struct {
			uint8_t isGameStateSaved : 1;
			uint8_t availableToUse : 7;
		};
	};

	uint8_t depth;
	uint8_t savedGameState[_Game::PACKED_SIZE];

#ifdef DBG_MODE
	vector<MCTS_Node*> tmpChildren;
#endif
	//Parent is ommitted, the MCTS Search keeps track of children to do the backpropagation
	MCTS_Node() {}
	virtual ~MCTS_Node() {}

	//selection: select child according to eval value + C * sqrt(log(parent visits) / visits), or if child is not visited: eval value + FPU, until you reach node leaf
	inline void Reset(MCTS_Node* _parent, Move m, float _policy);
	string printGraph(PRECACHE& c);
};

//Structures for MCTS Node caching. Instead of having a big array of MCTS Node I have a vector<vector<MCTS_Node>>. This structure is for simplifying tree reuse.
//I can mark what PRECACHE_PACKs are in use and clear the rest. Tree reuse can be harder on a big array (if you never fill it then it's not a problem)
struct PRECACHE_PACK {
	MCTS_Node cache[PRECACHE_NODECOUNT_PACK];
	int index = 0;
	int saveMark = 0;
	bool reserve(uint16_t& firstChild, const uint16_t& ChildCount) {
		ASSERT(ChildCount != 0 && ChildCount != NO_CHILDREN);
		if (index + (int)ChildCount < PRECACHE_NODECOUNT_PACK - 1)
		{
			firstChild = (uint16_t)index;
			index += (int)ChildCount;
			return true;
		}
		return false;
	}
	void clear() {
		index = 0;
	}
	PRECACHE_PACK() {}
};

struct PRECACHE {
	NodeIndex root;
	long long CacheNodeSizeBytes = 0;
	vector<PRECACHE_PACK*> cache_Node;

	int RolloutBlockID = 0;

	~PRECACHE() {
		for (auto& p : cache_Node)
		{
			delete[] p;
		}
		cache_Node.clear();
	}

	inline MCTS_Node* getNode(const NodeIndex& N) {
#ifdef DBG_MODE
		if (N.BlockID >= cache_Node.size())
		{
			cerr << "Block Error:" << N.BlockID << endl;
		}
		if (N.FirstChild >= PRECACHE_NODECOUNT_PACK)
		{
			cerr << "Child Error:" << N.FirstChild << endl;
		}
#endif
		return &cache_Node[N.BlockID]->cache[N.FirstChild];
	}

	void Init(long long  _CacheNodeSizeBytes) {
#ifndef _MSC_VER
		Stopwatch localTimer;
		localTimer.Start(300 * 1000);
#endif
		int TargetSize = (int)_CacheNodeSizeBytes / (int)(sizeof(PRECACHE_PACK));
		int CONS = 0;
		for (int i = 0; i < TargetSize; ++i) {
			PRECACHE_PACK* p = new (nothrow) PRECACHE_PACK;
			if (p != nullptr)
			{
				cache_Node.push_back(p);
			}
			else break; //Not enough Memory
#ifndef _MSC_VER
			if (localTimer.Timeout())
				break;
#endif
		}
		RolloutBlockID = (int)cache_Node.size() - 1;
		CacheNodeSizeBytes = (int)sizeof(PRECACHE_PACK) * (int)cache_Node.size();
		if (cache_Node.size() < 6)
		{
			cerr << "ERROR: NOT ENOUGH CACHE MEMORY. Nodepacks:" << cache_Node.size() << " Cache Memory:" << CacheNodeSizeBytes << endl;
		}
		else {
			cerr << "Cache Manager. Size:" << (CacheNodeSizeBytes / 1024 / 1024) << "MB - " << cache_Node.size() * PRECACHE_NODECOUNT_PACK << " nodes splitted in " << cache_Node.size() << " blocks. Rollout Block ID is " << RolloutBlockID << endl;
		}
	}

	void recursiveMark(MCTS_Node* N, const int& mark) {
		if (N->ChildCount == IS_LEAF || N->ChildCount == NO_CHILDREN)
			return;
		if (cache_Node[N->ChildIndex.BlockID]->saveMark != mark)
			cache_Node[N->ChildIndex.BlockID]->saveMark = mark;
		MCTS_Node* child = getNode(N->ChildIndex);
		for (int i = 0; i < N->ChildCount; ++i)
		{
			recursiveMark(child, mark);
			++child;
		}
	}
	void setRoot(int cacheIndex, int childIndex)
	{
		int mark = (rand() << 20) + rand();//childIndex * 10000 + cacheIndex * 10 + (rand() % 10);
		root.BlockID = cacheIndex;
		root.FirstChild = childIndex;
		cache_Node[root.BlockID]->saveMark = mark;
		MCTS_Node* N = getNode(root);
		recursiveMark(N, mark);
		int cleared = 0;
		for (auto& c : cache_Node)
		{
			if (c->saveMark != mark)
			{
				cleared++;
				c->clear();
			}
		}
		DBG(cerr << " New Root, cleared " << cleared << "/" << cache_Node.size() << " cache blocks. T:" << stopwatch.EllapsedMilliseconds() << "ms" << endl;);
	}

	void reserve(const MCTS_Node* parent, const int ChildCount, uint16_t& cacheIndex, uint16_t& firstChild)
	{
		int i = (parent == nullptr ? root.BlockID : parent->ChildIndex.BlockID) + 1;
		for (size_t ni = 0; ni < cache_Node.size(); ++ni)
		{
			if (i >= RolloutBlockID) //Ignore RolloutBlockID
			{
				i = 0;
			}
			if (cache_Node[i]->reserve(firstChild, ChildCount))
			{
				cacheIndex = i;
				return;
			}
			++i;
		}
		cerr << "FATAL ERROR, CAN'T RESERVE" << endl;
		abort();
	}

	void reserveRollout(const int ChildCount, uint16_t& cacheIndex, uint16_t& firstChild)
	{
		if (!cache_Node[RolloutBlockID]->reserve(firstChild, ChildCount))
		{
			cerr << "FATAL ERROR, CAN'T RESERVE ROLLOUT" << endl;
			abort();
		}
		cacheIndex = RolloutBlockID;
		return;
	}

	void clearRollouts() {
		cache_Node[RolloutBlockID]->clear();
	}
};

const float WORST_SCORE = -9999999999.99f;


string MCTS_Node::printGraph(PRECACHE& c) {
	auto nodo = this;
	string s = "";
	//for (auto& nodo : playedNodes)
	while (nodo != nullptr)
	{
		int childvisits = 0;
		if (nodo->ChildCount != NO_CHILDREN)
		{
			auto child = c.getNode(nodo->ChildIndex);

			int childvisits = 0;
			for (int i = 0; i < nodo->ChildCount; ++i)
			{
				childvisits += (child + i)->visits;
			}
		}
		s = "->" + to_string(nodo->ChildIndex.BlockID * 100000 + nodo->ChildIndex.FirstChild) + "{CvV:" + to_string(nodo->ChildCount) + "/" + to_string(childvisits) + "/" + to_string(nodo->visits) + "}" + s;
		nodo = nodo->parent;
	}

	//	s+= " Score:"+ to_string(playedNodes.back()->SumScore)+" "+to_string(playedNodes.back()->SumScore);
	return s;
}

inline void MCTS_Node::Reset(MCTS_Node* _parent, Move m, float _policy) {
	nnValue = 0.0f;
	sumScore = 0.0f;
	policy = _policy;
	visits = 0;
	action = m;
	ChildIndex.FirstChild = 0;
	parent = _parent;
	if (parent != nullptr)
	{
		ChildIndex.BlockID = parent->ChildIndex.BlockID;
		depth = parent->depth + 1;
	}
	else {
		ChildIndex.BlockID = 0;
		depth = 0;
	}
	maxScore = 0.0f;

	ChildCount = IS_LEAF;
	isGameStateSaved = 0;

	ChildIndex.FirstChild = 0;
	Status = 0;

#ifdef DBG_MODE
	tmpChildren.resize(0);
#endif

}

const int MAX_PLAYERS = 2;
const int MAX_PLAYER_UNITS = 1;

vector<Move> tmpMove(1);

//*************************************************** Modified MCTS for AlphaZero **************************************************************************//

class MCTS {
public:
	MCTS_Conf conf; //Parameters that control 
	PRECACHE cache; //New nodes will be requested to that precache
	Stopwatch* timecontrol = nullptr;
	Game lastTurn = nullptr;
	Random rnd;
	vector<Move> MKCHlist;
	int rolloutCount = 0;
	bool dontGenerateReplay = false;
	vector<MCTS_Node*> rootNodes;
	vector<MCTS_Node*> bestNodeToPlay;
	MCTS() { }//bogus
	MCTS(Stopwatch* st, int _CacheNodeSizeBytes) {
		conf = default_conf;
		timecontrol = st;
		cache.Init(_CacheNodeSizeBytes);
		if (conf.mode == mcts_mode::submit)
			cerr << " Cache Init " << st->EllapsedMilliseconds() << "ms" << endl;
	}
	MCTS(const MCTS_Conf& _conf, Stopwatch* st, int _CacheNodeSizeBytes) {
		conf = _conf;
		timecontrol = st;
		cache.Init(_CacheNodeSizeBytes);
		if (conf.mode == mcts_mode::submit)
			cerr << " Cache Init " << st->EllapsedMilliseconds() << "ms" << endl;
	}


	//Not used. I don't understand quite well the temperature thingy on AZ.
	//I understand that it changes from a normal policy output to a one-hot output (all zero except the selected action as 1.0).
	//But it's unclear to me how to tune it, so I don't use it.
	void apply_temperature(const Game& gamestate) {
		float temp = gamestate->getInitialTemperature();
		if (temp == 1.0f)
		{
			//Same policies
			return;
		}
		else if (temp <= 0.000001f)
		{
			// 0,0,0,1,0,0,0,0,0 policy. Only 1 value
			for (auto& rootNode : rootNodes)
			{
				auto node = cache.getNode(rootNode->ChildIndex);
				float maxVal = node->policy;
				int maxIndex = 0;
				if (rootNode->ChildCount != 0 && rootNode->ChildCount != NO_CHILDREN)
				{
					for (int i = 1; i < rootNode->ChildCount; ++i) {
						float tmpV = (node + i)->policy;
						if (tmpV > maxVal) {
							maxVal = tmpV;
							maxIndex = i;
						}
					}

					for (int i = 0; i < rootNode->ChildCount; ++i) {
						(node + i)->policy = (i == maxIndex ? 1.0f : 0.0f);
					}
				}
			}
		}
		else {
			for (auto& rootNode : rootNodes)
			{
				auto node = cache.getNode(rootNode->ChildIndex);
				float sumPol = 0.0f;
				if (rootNode->ChildCount != 0 && rootNode->ChildCount != NO_CHILDREN)
				{
					for (int i = 1; i < rootNode->ChildCount; ++i) {
						(node + i)->policy = pow((node + i)->policy, 1.0f / temp);
						sumPol += (node + i)->policy;
					}
					sumPol = 1.0f / sumPol;
					for (int i = 1; i < rootNode->ChildCount; ++i) {
						(node + i)->policy *= sumPol;
					}
				}

			}
		}
	}



	void printTree(MCTS_Node* N, int __depth, int __maxDepth, ostream* DEBUG_VIEW_TREE)
	{
		if (N->ChildCount == IS_LEAF)
			return;
		if (__depth > __maxDepth)
			return;
		*DEBUG_VIEW_TREE << std::string(__depth * 2, ' ');
		int sumChildVisits = N->ChildCount == IS_LEAF ? 0 : 1;
		float sumValues = N->ChildCount == IS_LEAF ? 0.0f : N->nnValue;

		if (N->ChildCount > 0 && N->ChildCount != NO_CHILDREN)
		{
			auto child = cache.getNode(N->ChildIndex);
			for (int i = 0; i < N->ChildCount; ++i)
			{
				sumChildVisits += child->visits;
				sumValues += -child->sumScore;
				++child;
			}
		}
		int ID = (N->parent == nullptr ? 0 : N->parent->ChildIndex.BlockID * 100000 + N->parent->ChildIndex.FirstChild + (int)(N - cache.getNode(N->parent->ChildIndex)));
		*DEBUG_VIEW_TREE << ID << ":D" << (int)N->depth << " N:" << N->visits << "|" << sumChildVisits << " Vn:" << N->nnValue <<" Max:"<<N->maxScore<< " Sum:" << N->sumScore << "|" << sumValues << " Mean:" << N->sumScore / (float)max(1, N->visits) << " Pol:" << N->policy << " Child:" << (int)N->ChildCount << " St:" << (int)N->Status << endl;
		if (N->ChildCount > 0 && N->ChildCount != NO_CHILDREN)
		{
			auto child = cache.getNode(N->ChildIndex);
			for (int i = 0; i < N->ChildCount; ++i)
			{
				printTree(child, __depth + 1, __maxDepth, DEBUG_VIEW_TREE);
				++child;
			}
		}

	}
	
	
	//Read a gamestate and prepare it for exporting as a sample.
	//
	ReplayMove getReplayBuffer(Model& model, Game& gamestate) {
		ReplayMove rm;

		//This variable ensures that if I randomly picked a move, these values won't be stored as a sample for that turn.
		rm.ignoreDontSave = dontGenerateReplay;
		rm.testGame.CopyFrom(gamestate);
		rm.currIDToPlay = gamestate->idToPlay;
		rm.tmpSwapped = gamestate->swapped;

		if (rm.currIDToPlay != 0) {
			gamestate->swapPlayers();
		}
		rm.tmpIDToPlay = gamestate->idToPlay;
		gamestate->setNNInputs(model, gamestate->idToPlay);
		auto& inputs = model.inputs[0]->output;
		float* tns = (float*)&inputs.xmm[0].v;
		rm.gamestate.resize(inputs.size);
		for (int i = 0; i < inputs.size; ++i) {
			rm.gamestate[i] = *tns;
			++tns;
		}

		rm.validMovesNr = rootNodes[0]->ChildCount != NO_CHILDREN ? rootNodes[0]->ChildCount : 0;
		//Current state value, it will be tweaked later with endgame 
		rm.meanScore = rootNodes[0]->sumScore / (float)max(1, rootNodes[0]->visits);
		rm.policy.resize(_Game::getPolicySize());
		Tensor* ou = &model.outputs[1]->output;
		tns = (float*)&ou->xmm[0];

		//Softmax set all the output tensor to zero
		for (int i = 0; i < (ou->xmm_size * 8); ++i)
		{
			*(tns + i) = -9999999.99f;
		}
		//Invalids are set as negative. Numpy can easily filter out. This was for some testing on Tensorflow about ignoring losses on invalid moves.
		//Se can set it as 0.0f
		fill(rm.policy.begin(), rm.policy.end(), -0.00001f);

		if (rm.validMovesNr > 0)
		{
			auto node = cache.getNode(rootNodes[0]->ChildIndex);
			float sumVisits = 0.0f;
			for (int i = 0; i < rm.validMovesNr; ++i)
				sumVisits += (float)(node + i)->visits;
			//Load visits
			for (int i = 0; i < rm.validMovesNr; ++i) {
				*(tns + ((node + i)->action)) = (float)(node + i)->visits / sumVisits;
			}
			//Do softmax on all the output
			if (REPLAY_SOFTMAX_POLICY)
				Activation_Softmax(*ou, *ou);
			//Pass as policy
			for (int i = 0; i < rm.validMovesNr; ++i) {
				int index = (int)(node + i)->action;
				rm.policy[index] = *(tns + index);
			}
		}


		if (rm.validMovesNr == 0)
			rm.selectedMove = 0;
		else rm.selectedMove = bestNodeToPlay[0]->action;

		if (rm.currIDToPlay != 0) {
			gamestate->swapPlayers();
		}

		return rm;
	}


	//Once the Search is completed (by rollout count or timeout) this function 
	void pickBestPlay(const Game& gamestate, vector<Move>& bestMove, ostream* DEBUG_VIEW_TREE = nullptr) {
		dontGenerateReplay = false;
		bestNodeToPlay.resize(0);
		bestMove.resize(0);
		//Update root node policies
		if (DEBUG_VIEW_TREE)
		{
			*DEBUG_VIEW_TREE << gamestate->Print() << endl;
		}
		float sumVisits = 0.0f;
		float sumScore = 0.0f;
		for (auto& rootNode : rootNodes)
		{
			//Ugly hack, I have a bug where turn 200 isn't giving any move.
			if (rootNode->ChildCount == NO_CHILDREN || rootNode->ChildCount == IS_LEAF)
			{
				
				vector<Move> pm;
				gamestate->getPossibleMoves(0, pm, 0);
				cerr << "ERROR: CHILDCOUNT!!!!"<<"Posibles:"<<pm.size() << endl;
				_Game cp;
				float bestScore = -99.0f;
				Move bestAction = 0;
				for (auto& m : pm)
				{
					cp.CopyFrom(gamestate);
					cp.Simulate(m);
					float tmpSC = cp.EvalPlayer(0, 0);
					if (tmpSC > bestScore)
					{
						bestScore = tmpSC;
						bestAction = m;
					}
				}
				cerr << " Failover to " << (int)bestAction << " SC:" << bestScore << endl;
				bestNodeToPlay.push_back(nullptr);
				bestMove.push_back(bestAction);
				return;
			}
			auto node = cache.getNode(rootNode->ChildIndex);
			for (int i = 0; i < rootNode->ChildCount; ++i) {
				sumVisits += (float)(node + i)->visits;
				sumScore += (float)(node + i)->sumScore;
			}
			rootNode->visits = (int)sumVisits;
			rootNode->sumScore = sumScore;
			sumVisits = 1.0f / sumVisits;

		}
		//Not sure about this. Disabled for now:	apply_temperature(gamestate);

		//Add more randomizations at first turns.
		//Before turn 12 I'll create random moves on selfplay, and I'll mark it as "dontGenerateReplay"
		//I think this doesn't compromise sample quality, because the "offending" selection won't appear in samples.
		if (conf.mode == mcts_mode::selfplay &&  gamestate->turn < 12 && rnd.NextInt(1000) < 15)
		{
			for (auto& rootNode : rootNodes)
			if (rootNode->ChildCount != IS_LEAF && rootNode->ChildCount != 0)
			{
				auto node = cache.getNode(rootNode->ChildIndex);
				int randomPick = rnd.NextInt(rootNode->ChildCount);
				MCTS_Node* mostVisited = node+ randomPick;
				bestNodeToPlay.push_back(mostVisited);
				bestMove.push_back(mostVisited->action);

				dontGenerateReplay = true;
				return;
			}
		}


		int Visits = 0;

		int tmpCheckVisits = 0;

		//Best Move selection. I'll pick all moves between a range . Any move with similar scores (95%/98% of the best score so far) will be taken into account.
		float COEF_LIMIT_BEST = (conf.mode == mcts_mode::selfplay ? 0.95f : 0.98f);
		vector< pair< MCTS_Node*,float>> goodMoves;
		for (auto& rootNode : rootNodes)
		{
			if (conf.mode == mcts_mode::submit)
				printTree(rootNode, 0, VIEW_TREE_DEPTH, &cerr);
			if (DEBUG_VIEW_TREE)
			{
				if (conf.mode != mcts_mode::submit)
					printTree(rootNode, 0, VIEW_TREE_DEPTH, DEBUG_VIEW_TREE);
			}

			auto node = cache.getNode(rootNode->ChildIndex);
			if (DEBUG_VIEW_TREE)
			{
				DebugSelect(rootNode, node, 0, gamestate->turn, DEBUG_VIEW_TREE);
			}
			MCTS_Node* mostVisited = node;
			float bestValue = -9999.99f;
			int nCount = rootNode->ChildCount;


			for (int i = 0; i < nCount; ++i) {
				Visits += node->visits;

				float v = (float)node->visits;
				float mean = node->sumScore / v;


				//Alternative move selection. Based on Jacek's idea. I uses both visits and mean score.
				float tmpValue = mean/*node->maxScore*/ + logf((float)(v + 3.0f));
				//float tmpValue = mean/*node->maxScore*/ + logf(sqrtf((float)(v + 3.0f))) * 0.1f; //Some tweaks to reduce visit importance
				if (DEBUG_VIEW_TREE)
				{
					*DEBUG_VIEW_TREE << node->visits << " " << node->policy << " -> " << ((float)node->visits) * sumVisits<<" JaceK:"<< mean<<"|"<< tmpValue-mean<<"|"<< tmpValue;
				}
				
				//Keep best score and those with 98% of its score
				if (tmpValue > COEF_LIMIT_BEST*bestValue)
				{
					if (tmpValue > bestValue)
					{
						bestValue = tmpValue;
						if (goodMoves.size()>0)
						for (int j = (int)goodMoves.size()-1; j >= 0; --j)
						{
							if (goodMoves[j].second <= COEF_LIMIT_BEST *bestValue)
							{
								goodMoves.erase(goodMoves.begin() + j);
							}
						}
					}
					goodMoves.emplace_back(pair < MCTS_Node*, float>(node, tmpValue));
					if (DEBUG_VIEW_TREE)
					{
						*DEBUG_VIEW_TREE << " *"<< tmpValue<<" > "<< COEF_LIMIT_BEST *bestValue;
					}
				}
				
				//This is the original Move selection, just a visits checks. I prefer to take into acount Mean Score.
			/*	if (node->visits > mostVisited->visits)
				{
					mostVisited = node;
					if (DEBUG_VIEW_TREE)
					{
						*DEBUG_VIEW_TREE << " *";
					}
				}*/
				if (DEBUG_VIEW_TREE)
				{
					_Game tGM;
					tGM.Unpack(node->savedGameState);
					*DEBUG_VIEW_TREE << " SC:" << (int)tGM.score0 << " " << (int)tGM.score1;
					*DEBUG_VIEW_TREE << endl;
				}
				node++;
			}
			if (DEBUG_VIEW_TREE) {
				cerr << "There are " << goodMoves.size() << " good moves:";
				for (auto&node : goodMoves)
				{
					cerr << node.second << ",";
				}
				cerr<< endl;
			}
			//Pick a random move between the best ones.
			mostVisited = goodMoves[ rnd.NextInt((uint32_t) goodMoves.size())].first;
			bestNodeToPlay.push_back(mostVisited);
			bestMove.push_back(mostVisited->action);
		}
		if (DEBUG_VIEW_TREE)
		{
			*DEBUG_VIEW_TREE << " Visits:" << rootNodes[0]->visits << "/" << tmpCheckVisits << "/" << Visits << " T:" << timecontrol->EllapsedMilliseconds() << "ms";
			*DEBUG_VIEW_TREE << endl;
		}
		DBG(if (conf.mode == mcts_mode::submit) cerr << " Visits:" << rootNodes[0]->visits << "/" << tmpCheckVisits << "/" << Visits << " T:" << timecontrol->EllapsedMilliseconds() << "ms" << endl;);
	}


	inline void backPropagate(MCTS_Node* leaf) {
		float score = leaf->nnValue;
		MCTS_Node* node = leaf;

		while (node != nullptr) {
			node->sumScore += score;
			++node->visits;
			//>>MAX SCORE Calculation. Not really used. I wanted to test JacekMax, but haven't tested it
			if (node->ChildCount != IS_LEAF && node->ChildCount != NO_CHILDREN)
			{
				float newMax = -9999.99f;
				MCTS_Node* child = cache.getNode(node->ChildIndex);
				for (int i = 0; i < node->ChildCount; ++i)
				if ((child + i)->ChildCount != IS_LEAF)
				{
					float childVal = -(child + i)->maxScore;
					if (childVal > newMax)
					{
						newMax = childVal;
					}

				}
				if (newMax > -1.5f)
				{
					node->maxScore = newMax;
				}
			}
			//<<MAX SCORE Calculation. 
			node = node->parent;
			score = -score;
		}
	}


	inline MCTS_Node* Select(MCTS_Node* parent, MCTS_Node* firstChild, int _depth, int _turn) {
		if (parent->ChildCount == 1)
		{
			return firstChild;
		}
		MCTS_Node* child = firstChild;//cache.getNode(CacheIndex);
		float turn_cpuct;
		//QUESTION ABOUT CPUCT: USE DEPTH, TURN OR VISITS?
		if (conf.cpuct_inc > 0.0f)
		{
			turn_cpuct = min(conf.cpuct_limit, conf.cpuct_base + conf.cpuct_inc * (float)_turn); 
		}
		else {
			turn_cpuct = max(conf.cpuct_limit, conf.cpuct_base + conf.cpuct_inc * (float)_turn);
		}
		float parent_F = (parent->visits <= 1 ? turn_cpuct : turn_cpuct * fastsqrtf((float)parent->visits));


		MCTS_Node* bestChild = child;
		float bestUCT = -9999999.0f;
		for (int i = 0; i < parent->ChildCount; ++i)
		{
			float Q = (child->visits <= 1 ? child->sumScore : child->sumScore * fastinv(child->visits));
			float J = child->maxScore;
			//const float JACEK_COEFF = 0.0f;
			//float P = iszero(ϵ) ? policy : (1-ϵ) * policy + ϵ * η[i]
			float U = parent_F * child->policy * fastinv(1 + child->visits);

			//float PUCT = JACEK_COEFF*J+(1.0f- JACEK_COEFF)*Q + U;
			float PUCT = Q + U;
			if (PUCT > bestUCT)
			{
				bestChild = child;
				bestUCT = PUCT;
			}
			++child;
		}
		return bestChild;
	}

	inline MCTS_Node* DebugSelect(MCTS_Node* parent, MCTS_Node* firstChild, int _depth, int _turn, ostream* DEBUG_VIEW_TREE) {
		if (parent->ChildCount == 1)
		{
			return firstChild;
		}
		MCTS_Node* child = firstChild;//cache.getNode(CacheIndex);
		float turn_cpuct;
		if (conf.cpuct_inc > 0.0f)
		{
			turn_cpuct = min(conf.cpuct_limit, conf.cpuct_base + conf.cpuct_inc * (float)_turn);
		}
		else {
			turn_cpuct = max(conf.cpuct_limit, conf.cpuct_base + conf.cpuct_inc * (float)_turn);
		}
		float parent_F = (parent->visits <= 1 ? turn_cpuct : turn_cpuct * fastsqrtf((float)parent->visits));
		*DEBUG_VIEW_TREE << "parent_F: CPUCT" << turn_cpuct << " SQRT:" << fastsqrtf((float)parent->visits) << " " << sqrtf((float)parent->visits) << " = " << parent_F << endl;
		MCTS_Node* bestChild = child;
		float bestUCT = -9999999.0f;
		for (int i = 0; i < parent->ChildCount; ++i)
		{
			float Q = (child->visits <= 1 ? child->sumScore : child->sumScore * fastinv(child->visits));
			//float P = iszero(ϵ) ? policy : (1-ϵ) * policy + ϵ * η[i]
			float U = parent_F * child->policy * fastinv(1 + child->visits);
			float PUCT = Q + U;
			*DEBUG_VIEW_TREE << "Child:" << i << " Visits:" << child->visits << " Val:" << child->sumScore << " " << child->sumScore / (float)(max(1, child->visits)) << " NNVal:" << child->nnValue << " Q:" << Q << " U:" << U << " = " << PUCT << endl;

			if (PUCT > bestUCT)
			{
				bestChild = child;
				bestUCT = PUCT;
			}
			++child;
		}
		return bestChild;
	}

	void dirichlet_noise(MCTS_Node* parent, float epsilon, float alpha) {
		if (epsilon < 0.001f) //no epsilon
			return;
		if (alpha < 0.001f)
			return;
		random_device rd;
		mt19937 gen(rd());
		vector<float> dirichlet_vector(parent->ChildCount);
		gamma_distribution<float> gamma(alpha, 1.0f);


		float factorDirich = 0.0f;
		for (int i = 0; i < parent->ChildCount; i++) {
			dirichlet_vector[i] = gamma(gen);
			factorDirich += dirichlet_vector[i];
		}

		if (factorDirich < numeric_limits<float>::min()) {
			return;
		}
		factorDirich = epsilon / factorDirich;

		auto child = cache.getNode(parent->ChildIndex);
		for (int i = 0; i < parent->ChildCount; i++)
		{
			child->policy = child->policy * (1.0f - epsilon) + dirichlet_vector[i] * factorDirich;
			++child;
		}
	}


	_Game tmpGScalc;

	inline int Expand(Model& model, MCTS_Node* parent, const int& _depth, Game& working) {
		int ownerID = working->getIDToPlay();
		working->getPossibleMoves(0, MKCHlist, _depth);
		int childCount = (int)MKCHlist.size();
		parent->ChildCount = (uint8_t)childCount;
#ifdef DBG_MODE
		parent->tmpChildren.resize(0);
#endif
		if (childCount == 0 || working->isEndGame())
		{  //I assume it's a killed unit -> reached endgame 
			parent->ChildCount = NO_CHILDREN;
			//Add endgame value, from some heuristic and not from the NN
			parent->nnValue = working->EvalPlayer(1 - ownerID, _depth);
			parent->maxScore = parent->nnValue;
			return 0;
		}
		else {
			//Create children
			cache.reserve(parent, childCount, parent->ChildIndex.BlockID, parent->ChildIndex.FirstChild);
			//Run NN, get Policy and Value
			Tensor* policy = &model.outputs[1]->output;
			MCTS_Node* child = cache.getNode(parent->ChildIndex);

			if (!conf.useHeuristicNN)
			{
				working->predict(model, ownerID, &policy, parent->nnValue);
			}
			else {
				abort(); //Removed
			}
			if (parent->parent == nullptr)
			{
				parent->nnValue = -parent->nnValue;
			}

			if (conf.simpleRandomRange > 0.0f)
			{
				float randNoise = 1.0f + rnd.NextFloat(-conf.simpleRandomRange, conf.simpleRandomRange);
				parent->nnValue *= randNoise;
			}
			parent->maxScore = parent->nnValue;
			//Create children, no visits yet


			if (childCount == 1)
			{
				child->Reset(parent, MKCHlist[0], 1.0f);
			}
			else
			{
				float sumPolicy = 0.0f;

				//Removing invalid moves
#ifdef REMOVE_SOFTMAX_INVALID_MOVES
				if (!conf.useHeuristicNN)
				{
					uint64_t maskValid = 0ULL;
					for (auto& cc : MKCHlist) {
						maskValid |= (1ULL << cc);
					}

					for (int i = 0; i < 8; ++i)
					{
						if ((maskValid & (1ULL << i)) == 0ULL)
						{
							policy->setElement(i, -999999999.99f);
						}
					}
					Activation_Softmax(*policy, *policy);
				}
#endif
				for (int i = 0; i < childCount; ++i)
				{
					(child + i)->Reset(parent, MKCHlist[i], 0.0f);
					float child_pol;

					if (!conf.useHeuristicNN)
					{
						child_pol = policy->getElement((uint32_t)MKCHlist[i]);
					}
					else {
						abort(); //removed
					}
					(child + i)->policy = child_pol;
					sumPolicy += child_pol;
				}
				if (conf.useHeuristicNN)
				{ //Policy normalization
					abort(); //Removed
				}

				if (parent->parent == nullptr && _depth == 0 && conf.dirichlet_noise_epsilon > 0.0f)
				{
					dirichlet_noise(parent, conf.dirichlet_noise_epsilon - conf.dirichlet_decay * (float)working->turn, conf.dirichlet_noise_alpha);
				}
			}
		}
		return childCount;
	}
	//Tree Reuse - Recover new root from current tree
	void RestoreRoot(const Game& gamestate) {
		bool newRootFound = false;
		if (conf.mode == mcts_mode::submit && lastTurn != nullptr && bestNodeToPlay[0] != nullptr) //Search a depth=2 coherent GameState
		{
			vector<Move> mlist{ 0 };
#if CFG_Game_IsSavedInMCTSNode == 0
			Game move0 = lastTurn->Clone();
			mlist[0] = bestNodeToPlay[0]->action;
			move0->Simulate(mlist);
#endif
			//cerr << "Search Root...." << endl;
			auto& myPlay = bestNodeToPlay[0];
			MCTS_Node* enemyPlay = cache.getNode(myPlay->ChildIndex);
			Game move1 = make_shared<_Game>();
			for (int i = 0; i < myPlay->ChildCount; ++i) {
#if CFG_Game_IsSavedInMCTSNode == 1
				move1->Unpack(enemyPlay->savedGameState);
#else
				move1->CopyFrom(move0);
				mlist[0] = enemyPlay->action;
				move1->Simulate(mlist);
#endif
				//cerr << move1->Print() << endl;
				if (move1->Equals(gamestate))
				{
					if (enemyPlay->ChildCount == IS_LEAF || enemyPlay->ChildCount == NO_CHILDREN)
						break; //Bad thing
					 //Calc how many nodes we reused
					cerr << "NEW ROOT AT BLOCK:" << myPlay->ChildIndex.BlockID << " INDEXNODE:" << myPlay->ChildIndex.FirstChild + i << " FOUND. Visits:"
						<< enemyPlay->visits << "/" << rootNodes[0]->visits << " = " << (enemyPlay->visits * 100 / max(1, rootNodes[0]->visits)) << "% ";

					//We get the new root
					rootNodes[0] = enemyPlay;
					rootNodes[0]->parent = nullptr;
					newRootFound = true;
					cache.setRoot(myPlay->ChildIndex.BlockID, myPlay->ChildIndex.FirstChild + i);
					break;
				}
				++enemyPlay;
			}
		}
		if (!newRootFound)
		{

			if (gamestate->turn > 1 && conf.mode == mcts_mode::submit)
				cerr << "NOT FOUND " << stopwatch.EllapsedMilliseconds() << "ms" << endl;
			//Clear ALL
			for (auto& n : cache.cache_Node)
			{
				n->clear();
			}

			cache.root.BlockID = 0;
			cache.root.FirstChild = 0;
			cache.cache_Node[0]->index = 2;

			rootNodes[0] = cache.getNode(cache.root);
			rootNodes[0]->Reset(nullptr, 0, 0.0f);
			if (gamestate->turn > 1 && conf.mode == mcts_mode::submit)
				cerr << "RESET MOVES " << rootNodes[0] << "(" << (int)rootNodes[0]->ChildCount << "):";
			gamestate->Pack(rootNodes[0]->savedGameState);
			rootNodes[0]->isGameStateSaved = 1;
		}
		else {
			//	cerr << "RESTORED MOVES (" << (int)rootNodes[0]->ChildCount << "):[";
			if (rootNodes[0]->ChildCount != IS_LEAF && rootNodes[0]->ChildCount != NO_CHILDREN)
			{
				MCTS_Node* r = cache.getNode(rootNodes[0]->ChildIndex);
				for (int i = 0; i < rootNodes[0]->ChildCount; ++i)
				{
					//cerr << PrintMove(r->action,gamestate) << ",";
					++r;
				}
			}
			//cerr << "]" << endl;
		}
	}


	int maxDepth = 0;
	//Classic MCTS - Turn based games, 2 players
	int Search(Model& model, const Game& gamestate, vector<Move>& bestMove, ostream* DEBUG_VIEW_TREE = nullptr) {
		rolloutCount = 0;
		//Space reservation
		if (rootNodes.size() != 1) rootNodes.resize(1);
		if (bestNodeToPlay.size() != 1) bestNodeToPlay.resize(1);
		RestoreRoot(gamestate);

		lastTurn = gamestate->Clone();
		Game working = gamestate->Clone();

		MCTS_Node* current = rootNodes[0];
		vector<Move> tmpMoveList;
		tmpMoveList.resize(1);

		while ((!conf.useTimer && rolloutCount < conf.num_iters_per_turn) || (conf.useTimer && !timecontrol->Timeout()))
		{
			int _depth = 0;
			current = rootNodes[0];
			working->CopyFrom(gamestate);
			//Tree traverse until leaf, no simulation 
			while (current->ChildCount != IS_LEAF && current->ChildCount != NO_CHILDREN && !working->isEndGame())
			{
				current = Select(current, cache.getNode(current->ChildIndex), _depth, working->turn);

				++_depth;
#if CFG_Game_IsSavedInMCTSNode == 0
				tmpMoveList[0] = current->action;
				working->Simulate(tmpMoveList);
#endif
			}

			//On leaves calculate its value, make children and add policies to each.
			if (current->ChildCount == IS_LEAF)
			{
#if CFG_Game_IsSavedInMCTSNode == 1
				//recover the stored simulation
				if (current->parent != nullptr && current->isGameStateSaved == 0)
					working->Unpack(current->parent->savedGameState);
#endif	
				//Simulate
				if (current->parent != nullptr) {
					if (current->isGameStateSaved == 1)
					{
						working->Unpack(current->savedGameState);
					}
					else {
						working->Simulate(current->action);
#if CFG_Game_IsSavedInMCTSNode == 1
						//Save the stored simulation
						working->Pack(current->savedGameState);
						current->isGameStateSaved = 1;
#endif
					}
				}
				Expand(model, current, _depth, working);
				++_depth;

			}
			backPropagate(current);
			maxDepth = max(maxDepth, 1 + _depth);

			++rolloutCount;
		}
		DBG(if (conf.mode == mcts_mode::submit) cerr << "End Search, Max Depth:" << maxDepth << " Rollout Count:" << rolloutCount << endl;);
		pickBestPlay(gamestate, bestMove, DEBUG_VIEW_TREE);

		working->CopyFrom(gamestate);

		return 0;
	}


};



void _Game::setNNInputs(Model& model, const int& playerID) {
	//Neural Network inputs are always mirrored as player 0.
	auto& input = model.inputs[0]->output; //Get input reference
	
	float valueZero = 0.0f; //Instead of zero?
	float valueOne = 1.0f; //Instead of one?
	//clear;
	for (int i = 0; i < input.xmm_size; ++i)
	{
		input.xmm[i].v = _mm256_set1_ps(valueZero);//_mm256_setzero_ps();
	}
	int off0 = (playerID == 0 ? 0 : 6 * 24);
	int off1 = (playerID == 0 ? 6 * 24 : 0);


	float* tns = (float*)&input.xmm[0].v;

	//One-hot encoding, it seems to work better
	for (int i = 0; i < 6; ++i)
	{
		*(tns + off0 + (24 * i) + (cell0[i] > 23 ? 23 : cell0[i])) = valueOne;
		*(tns + off1 + (24 * i) + (cell1[i] > 23 ? 23 : cell1[i])) = valueOne;
	}
	off0 = 12 * 24 + (playerID == 0 ? 0 : 1 * 27);
	off1 = 12 * 24 + (playerID == 0 ? 1 * 27 : 0);
	*(tns + off0 + (score0 > 26 ? 26 : score0)) = valueOne;
	*(tns + off1 + (score1 > 26 ? 26 : score1)) = valueOne;
}



void _Game::predict(Model& model, const int& playerID, Tensor** policy, float& nVal)
{
	uint64_t HS;
	CachedNNEval* cEval = nullptr;

	//Recover NN Cache.
	if (USE_NNEVAL_CACHE)
	{
		HS = CalcHash(playerID);
		if (THREADS > 1)
			mutex_NNCache.lock();
		++NNCACHE_TOTAL;
		auto getCache = cacheNNEval.find(HS);
		if (getCache != cacheNNEval.end())
		{
			cEval = &getCache->second;
		}
	}
	//NN Eval not yet in cache, calculate
	if (cEval == nullptr)
	{
		if (USE_NNEVAL_CACHE && THREADS > 1)
			mutex_NNCache.unlock();
		//NOTE: I had soooo many problems with the signs, I won't touch anything because it seems to work right now.
		setNNInputs(model, playerID);
		model.predict();
		*policy = &model.outputs[1]->output;
		//Set that value as Score;
		nVal = -model.outputs[0]->output.getElement(0);
		//NOTE: This failed.....
		//if (idToPlay == 0)
			//nVal = -nVal;

		//Send to cache.
		if (USE_NNEVAL_CACHE)
		{
			if (THREADS > 1)
				mutex_NNCache.lock();
			cacheNNEval[HS].output = model.outputs[1]->output;
			cacheNNEval[HS].nVal = nVal;
			++NNCACHE_MISS;
		}
	}
	else {
		model.outputs[1]->output = cEval->output;
		*policy = &model.outputs[1]->output;
		nVal = cEval->nVal;
		++NNCACHE_HIT;
	}
	if (USE_NNEVAL_CACHE && THREADS > 1)
		mutex_NNCache.unlock();
}

float _Game::EvalNN(Model& model, const int& playerID, const int& _depth)
{
	Tensor* policy;
	float val;
	predict(model, playerID, &policy, val);
	return (playerID == 0 ? val : -val);
}

float _Game::EvalHeuristic(const int& playerID, const int& _depth)
{
	float Score = 0.0f;
	//Disabled, not in use
	if (Score == 0.0f)
		abort();
	return Score;
}


Random sc;
float _Game::EvalPlayer(const int& playerID, const int& _depth) {
	float Score = 0.000000001f;
	if (isEndGame())
	{
		int winner = getWinnerID();
		if (winner == 1)
		{
			Score = -0.85f //BASE_WIN_SCORE
				- (float)(MAX_TURNS - turn) * 0.0006f //Estimated range 0.85 to 0.97
				- 0.0026f * (float)score1
				+ 0.0020f * (float)score0;
		}
		else if (winner == 0)
		{
			Score = 0.85f //BASE_WIN_SCORE
				+ (float)(MAX_TURNS - turn) * 0.0006f //Estimated range 0.85 to 0.97
				+0.0026f * (float)score0
				-0.0020f * (float)score1;
		}
		else { //Draw
			Score = 0.0f + 0.0005f * (float)score0;
		}
	}
	else {
		Score = EvalHeuristic(playerID, _depth);
		Score *= sc.NextFloat(0.95f, 1.05f);
	}


	return (playerID == 0 ? Score : -Score);
}

/*********************************************** PIT WORKER - Match 2 models and get a winrate***************************************************************/
atomic<int> Pit_V1;
atomic<int> Pit_V2;
atomic<int> Pit_Draw;
atomic<int> matches;
//One worker per thread. Uses <atomic> to avoid race conditions
void Worker_Pit(int ID, string fileModel1, string fileModel2, int matchperWorker, MCTS_Conf conf1, MCTS_Conf conf2)
{
	//Each worker creates 2 independent MCTS trees
	Model candidateModel = _Game::CreateNNModel();
	candidateModel.loadWeights(fileModel1);
	Model currentModel = _Game::CreateNNModel();
	currentModel.loadWeights(fileModel2);
	if (!candidateModel.Loaded)
	{
		cerr << "Can't load model " << fileModel1 << endl;
		abort();
	}
	if (!currentModel.Loaded)
	{
		cerr << "Can't load model " << fileModel2 << endl;
		abort();
	}
	Game ws = make_shared<_Game>();
	MCTS* player1 = new MCTS(conf1, &stopwatch, 80 * 1024 * 1024);
	MCTS* player2 = new MCTS(conf2, &stopwatch, 80 * 1024 * 1024);
	vector<Move> bestMove;
	vector<Move> readMoves;
	for (int i = 0; i < matchperWorker; ++i) {
		//if (i%20 == randi)
		ws->Reset();
		ws->turn = 0;
		//Play alternatively as player0 or player1
		Model* m0 = ((i & 1) == 0) ? &candidateModel : &currentModel;
		Model* m1 = (m0 == &candidateModel ? &currentModel : &candidateModel);
		MCTS* p0 = ((i & 1) == 0) ? player1 : player2;
		MCTS* p1 = (p0 == player1 ? player2 : player1);
		p0->maxDepth = 0;
		p1->maxDepth = 0;
		while (true)
		{
			//Play as p0
			stopwatch.Start(45 * 1000);
			p0->Search(*m0, ws, bestMove);
			ws->Simulate(bestMove);
			if (!ws->isEndGame()) {
				ws->getPossibleMoves(0, readMoves, 0);//To force endgames
			}
			if (ws->isEndGame())
			{
				break;
			}
			//Play as P1
			ws->swapPlayers();
			stopwatch.Start(45 * 1000);
			p1->Search(*m1, ws, bestMove);
			ws->Simulate(bestMove);
			ws->swapPlayers();
			if (!ws->isEndGame())
			{
				ws->getPossibleMoves(0, readMoves, 0);//To force endgames
			}
			if (ws->isEndGame())
			{
				break;
			}
		}

		int winner = ws->getWinnerID();
		if (winner == 2)
		{
			++Pit_Draw;
		}
		else {
			if (m0 != &candidateModel) //Swap
			{
				winner = 1 - winner;
			}
			if (winner == 0)
				++Pit_V1;
			else ++Pit_V2;
		}
		{ //notify
			int totalGames = Pit_V1 + Pit_V2 + Pit_Draw;
			float winrate = 100.0f * ((float)Pit_V1 + 0.5f * (float)Pit_Draw) / (float)(totalGames);
			cerr << "Worker " << ID << ": " << Pit_V1 << "/" << Pit_V2 << "/" << Pit_Draw << ":" << winrate << "%";
			cerr << " NNCache:" << NNCACHE_HIT << "/" << NNCACHE_TOTAL << "/" << 100 * (NNCACHE_HIT) / (1 + NNCACHE_TOTAL) << "%";
			cerr<< endl;
		}

		++matches;
	}
}

//Read inputs, create <THREADS> Pit workers and then save the winrate on a file
int pitPlay(int argc, char* argv[])
{
	Pit_V1 = 0;
	Pit_V2 = 0;
	Pit_Draw = 0;
	matches = 0;
	int agc = 2;
	THREADS = atoi(argv[agc++]);
	int matchCount = atoi(argv[agc++]);
	int matchperWorker = matchCount / THREADS;

	string fileModel1 = string(argv[agc++]);
	MCTS_Conf conf1 = selfPlay_Mode;
	if (argc > agc) conf1.cpuct_base = (float)atof(argv[agc++]);
	if (argc > agc) conf1.cpuct_inc = (float)atof(argv[agc++]);
	if (argc > agc) conf1.cpuct_limit = (float)atof(argv[agc++]);
	if (argc > agc) conf1.num_iters_per_turn = atoi(argv[agc++]);
	if (argc > agc) conf1.dirichlet_noise_epsilon = (float)atof(argv[agc++]);
	if (argc > agc) conf1.dirichlet_noise_alpha = (float)atof(argv[agc++]);
	if (argc > agc) conf1.dirichlet_decay = (float)atof(argv[agc++]);
	if (argc > agc) conf1.simpleRandomRange = (float)atof(argv[agc++]);
	if (argc > agc) conf1.PROPAGATE_BASE = (float)atof(argv[agc++]);
	if (argc > agc) conf1.PROPAGATE_INC = (float)atof(argv[agc++]);
	if (argc > agc) conf1.POLICY_BACKP_FIRST = atoi(argv[agc++]);
	if (argc > agc) conf1.POLICY_BACKP_LAST = atoi(argv[agc++]);

	string fileModel2 = string(argv[agc++]);
	MCTS_Conf conf2 = conf1;
	if (argc > agc) conf2.cpuct_base = (float)atof(argv[agc++]);
	if (argc > agc) conf2.cpuct_inc = (float)atof(argv[agc++]);
	if (argc > agc) conf2.cpuct_limit = (float)atof(argv[agc++]);
	if (argc > agc) conf2.num_iters_per_turn = atoi(argv[agc++]);
	if (argc > agc) conf2.dirichlet_noise_epsilon = (float)atof(argv[agc++]);
	if (argc > agc) conf2.dirichlet_noise_alpha = (float)atof(argv[agc++]);
	if (argc > agc) conf2.dirichlet_decay = (float)atof(argv[agc++]);
	if (argc > agc) conf2.simpleRandomRange = (float)atof(argv[agc++]);
	if (argc > agc) conf2.PROPAGATE_BASE = (float)atof(argv[agc++]);
	if (argc > agc) conf2.PROPAGATE_INC = (float)atof(argv[agc++]);
	if (argc > agc) conf2.POLICY_BACKP_FIRST = atoi(argv[agc++]);
	if (argc > agc) conf2.POLICY_BACKP_LAST = atoi(argv[agc++]);

	cerr << "pitplay m1:" << fileModel1 << " m2:" << fileModel2 << " Th:" << THREADS << " N:" << matchCount << " C1:" << conf1.print() << " C2:" << conf2.print() << endl;
	vector<thread> threads(max(1, THREADS));
	for (int i = 0; i < max(1, THREADS); i++)
	{
		threads[i] = thread(Worker_Pit, i, fileModel1 + ".w32", fileModel2 + ".w32", matchperWorker, conf1, conf2);
	}

	for (int i = 0; i < max(1, THREADS); i++)
	{
		threads[i].join();
	}
	int totalGames = Pit_V1 + Pit_V2 + Pit_Draw;
	float winrate = 100.0f * ((float)Pit_V1 + 0.5f * (float)Pit_Draw) / (float)(totalGames);

	string PitFile = "./pitresults/Pit_" + fileModel1 + "_" + fileModel2 + "_" + to_string(winrate) + ".txt";
	ofstream f(PitFile);
	if (f.good())
	{
		f << winrate;
		f.close();
	}
	cout << winrate << endl;
	return 0;
}

/*********************************************** SELF-PLAY WORKER - Match 2 models and save samples from the match***************************************************************/

ReplayBuffer selfGames;
//One worker per thread. Uses mutexes to avoid race conditions
void Worker_SelfPlay(int ID, string fileModel1, string fileModel2, int matchperWorker, MCTS_Conf conf1, MCTS_Conf conf2) {
	cerr << "Worker " << ID << " will play " << matchperWorker << " matches" << endl;
	Model candidateModel = _Game::CreateNNModel();
	candidateModel.loadWeights(fileModel1);
	Model currentModel = _Game::CreateNNModel();
	currentModel.loadWeights(fileModel2);
	if (!candidateModel.Loaded)
	{
		cerr << "Can't load model " << fileModel1 << endl;
		abort();
	}
	if (!currentModel.Loaded)
	{
		cerr << "Can't load model " << fileModel2 << endl;
		abort();
	}
	Game ws = make_shared<_Game>();
	MCTS* player1 = new MCTS(conf1, &stopwatch, 80 * 1024 * 1024);
	MCTS* player2 = new MCTS(conf2, &stopwatch, 80 * 1024 * 1024);
	vector<Move> bestMove;
	vector<Move> readMoves;
	vector<float> backPropPolicy;
	backPropPolicy.resize(_Game::getPolicySize());
	ReplayGame RG;
	for (int i = 0; i < matchperWorker; ++i) {
		ws->Reset();
		ws->turn = 0;
		Model* m0 = ((i & 1) == 0) ? &candidateModel : &currentModel;
		Model* m1 = (m0 == &candidateModel ? &currentModel : &candidateModel);
		MCTS* p0 = ((i & 1) == 0) ? player1 : player2;
		MCTS* p1 = (p0 == player1 ? player2 : player1);
		p0->maxDepth = 0;
		p1->maxDepth = 0;

		RG.moves.resize(0);
		RG.reward = 0.0;

		int step = 0;

		while (true)
		{
			//Play as p0
			p0->Search(*m0, ws, bestMove, nullptr);
			RG.moves.emplace_back(p0->getReplayBuffer(*m0, ws));

            ws->Simulate(bestMove);
			if (!ws->isEndGame()) {
				ws->getPossibleMoves(0, readMoves, 0);//To force endgames
			}
			if (ws->isEndGame())
			{
				break;
			}
			//Play as P1
			ws->swapPlayers();
			p1->Search(*m1, ws, bestMove, nullptr);
			RG.moves.emplace_back(p1->getReplayBuffer(*m1, ws));

			ws->Simulate(bestMove);
			ws->swapPlayers();
			if (!ws->isEndGame())
			{
				ws->getPossibleMoves(0, readMoves, 0);//To force endgames
			}
			if (ws->isEndGame())
			{
				break;
			}
		}
		++matches;
		if (ws->swapped)
			ws->swapPlayers();
		RG.game = *ws;
		RG.reward = ws->EvalPlayer(0, 0);
		cerr << "Worker " << ID << ":" << matches << " W:" << RG.reward << " " << (int)ws->score0 << " " << (int)ws->score1 << " T:" << (int)ws->turn << "/" << RG.moves.size() << " Sw:" << (int)ws->swapped << " MAX DEPTH:p1:" << p0->maxDepth << "/p2:" << p1->maxDepth;
		cerr << " NNCache:" << NNCACHE_HIT << "/" << NNCACHE_TOTAL << "/" << 100 * (NNCACHE_HIT) / (1+NNCACHE_TOTAL) << "%";
		cerr<< endl;
		int totalMovesInReplay = 0;
		for (auto& r : RG.moves) {
			totalMovesInReplay += r.validMovesNr;
		}

		//Backpropagate policy. Similar to temperature on Alphazero
		//First 30% no backpropagate
		//Last 10% only selected policy
		int indexWithoutBackPolicy = (int)RG.moves.size() * conf1.POLICY_BACKP_FIRST / 100;
		int indexFullBackPolicy = (int)RG.moves.size() * (100 - conf1.POLICY_BACKP_LAST) / 100;
		for (int d = 0; d < (int)RG.moves.size(); ++d)
		{
			backPropPolicy = RG.moves[d].policy;
			for (auto& b : backPropPolicy)
			{
				if (b > 0.0f)
					b = 0.0f;
			}

			RG.moves[d].originalpolicy = RG.moves[d].policy;
			if (RG.moves[d].policy[RG.moves[d].selectedMove] > 0.0f) //Only if valid
			{
				backPropPolicy[RG.moves[d].selectedMove] = 1.0f;

				if (d <= indexWithoutBackPolicy)
				{
					RG.moves[d].factorBPP = 0.0f;
					//No backprop Policy
				}
				else if (d >= indexFullBackPolicy)
				{ //Only the selected move as policy
					RG.moves[d].policy = backPropPolicy;
					RG.moves[d].factorBPP = 1.0f;
				}
				else {
					//Linear conversion
					float factorBPP = (float)(d - indexWithoutBackPolicy) / (float)(indexFullBackPolicy - indexWithoutBackPolicy);
					RG.moves[d].factorBPP = factorBPP;
					for (int i = 0; i < (int)backPropPolicy.size(); ++i)
					{
						RG.moves[d].policy[i] = RG.moves[d].policy[i] * (1.0f - factorBPP) + factorBPP * backPropPolicy[i];
					}

				}
			}

		}

		//Backpropagate reward
		int accMoves = 0;
		float sReward = RG.reward;
		for (auto& r : RG.moves)
		{
			float rewardFactor = conf1.PROPAGATE_BASE + conf1.PROPAGATE_INC * ((float)accMoves / (float)totalMovesInReplay);

			r.originalValue = r.meanScore;
			r.backtrackValue = sReward;

			r.meanScore = rewardFactor * sReward + (1.0f - rewardFactor) * r.meanScore;
			sReward = -sReward;
			accMoves += r.validMovesNr;
		}

		//Dump data
		mutex_selfGames.lock();
		selfGames.games.emplace_back(RG);
		mutex_selfGames.unlock();

	}

}


//Read inputs, create <THREADS> Self-play workers and then save the winrate on a file
int selfPlay(int argc, char* argv[])
{
	matches = 0;
	int agc = 2;
	//Read command line parameters
	THREADS = atoi(argv[agc++]);
	int matchCount = atoi(argv[agc++]);
	int matchperWorker = matchCount / THREADS;

	string fileModel1 = string(argv[agc++]);
	MCTS_Conf conf1 = selfPlay_Mode;
	if (argc > agc) conf1.cpuct_base = (float)atof(argv[agc++]);
	if (argc > agc) conf1.cpuct_inc = (float)atof(argv[agc++]);
	if (argc > agc) conf1.cpuct_limit = (float)atof(argv[agc++]);
	if (argc > agc) conf1.num_iters_per_turn = atoi(argv[agc++]);
	if (argc > agc) conf1.dirichlet_noise_epsilon = (float)atof(argv[agc++]);
	if (argc > agc) conf1.dirichlet_noise_alpha = (float)atof(argv[agc++]);
	if (argc > agc) conf1.dirichlet_decay = (float)atof(argv[agc++]);
	if (argc > agc) conf1.simpleRandomRange = (float)atof(argv[agc++]);
	if (argc > agc) conf1.PROPAGATE_BASE = (float)atof(argv[agc++]);
	if (argc > agc) conf1.PROPAGATE_INC = (float)atof(argv[agc++]);
	if (argc > agc) conf1.POLICY_BACKP_FIRST = atoi(argv[agc++]);
	if (argc > agc) conf1.POLICY_BACKP_LAST = atoi(argv[agc++]);

	string fileModel2 = string(argv[agc++]);
	MCTS_Conf conf2 = conf1;
	if (argc > agc) conf2.cpuct_base = (float)atof(argv[agc++]);
	if (argc > agc) conf2.cpuct_inc = (float)atof(argv[agc++]);
	if (argc > agc) conf2.cpuct_limit = (float)atof(argv[agc++]);
	if (argc > agc) conf2.num_iters_per_turn = atoi(argv[agc++]);
	if (argc > agc) conf2.dirichlet_noise_epsilon = (float)atof(argv[agc++]);
	if (argc > agc) conf2.dirichlet_noise_alpha = (float)atof(argv[agc++]);
	if (argc > agc) conf2.dirichlet_decay = (float)atof(argv[agc++]);
	if (argc > agc) conf2.simpleRandomRange = (float)atof(argv[agc++]);
	if (argc > agc) conf2.PROPAGATE_BASE = (float)atof(argv[agc++]);
	if (argc > agc) conf2.PROPAGATE_INC = (float)atof(argv[agc++]);
	if (argc > agc) conf2.POLICY_BACKP_FIRST = atoi(argv[agc++]);
	if (argc > agc) conf2.POLICY_BACKP_LAST = atoi(argv[agc++]);

	//Prepare destination file
	string samplesFile = "./traindata/Replay_" + fileModel1 + "_" + fileModel2 + ".dat";
	//If it already exists, read all samples because it will be updated.
	processSamplesFile(samplesFile, _Game::getInputDimensions(), _Game::getPolicySize() + 1); //policy + value as output
	cerr << "selfplay m1:" << fileModel1 << " m2:" << fileModel2 << " Th:" << THREADS << " N:" << matchCount << " C1:" << conf1.print() << " C2:" << conf2.print() << endl;
	selfGames.games.resize(0);
	selfGames.games.reserve(matchCount);

	//Threading, each worker is independent, but all will add moves to the same Replay Buffer (using mutex to avoid race conditions)
	vector<thread> threads(max(1, THREADS));
	for (int i = 0; i < max(1, THREADS); i++)
	{
		threads[i] = thread(Worker_SelfPlay, i, fileModel1 + ".w32", fileModel2 + ".w32", matchperWorker, conf1, conf2);
	}

	for (int i = 0; i < max(1, THREADS); i++)
	{
		threads[i].join();
	}

	//The replay buffer must be deduplicated with existing samples. If a gamestate appears in two samples, it will sum the policy and value, and N will be increased.
	mutex_selfGames.lock();
	SamplesFile* sFile = getSampleFile(samplesFile);
	for (auto&G : selfGames.games)
	{
		//Backpropagated endgame value in turns before an ignoreDontSave might not be correct
		//Maybe a player win because the opponent did a random move on a critical point
		//So it's safer to just ignore those previous samples.
		bool ign = false;
		if (G.moves.size() > 1)
		for (int i=(int)G.moves.size()-1;i>=0;--i){
			if (G.moves[i].ignoreDontSave)
			{
				ign = true;
			}
			else if (ign)
			{
				G.moves[i].ignoreDontSave = true;
			}
		}		
		for (auto& R : G.moves)
		{
			if (R.ignoreDontSave)
				continue;
			//Convert the Replay Buffer to a SampleInfo. I did this way because previously I just saved all samples individually. This took too much disk space, was much slower.
			SampleInfo S;
			S.N = 1;
			S.I = R.gamestate;
			S.P = R.policy;
			S.P.emplace_back(R.meanScore);
			S.win = 0;
			S.loss = 0;
			S.draw = 0;
			if (S.P.back() > 0.45f)
			{
				++S.win;
			}
			else if (S.P.back() < -0.45f)
			{
				++S.loss;
			}
			else ++S.draw;
			//Insert sample to existing ones, deduplicating them.
			insertNewSample(sFile, S);
		}
	}
	//Store in file
	saveSamplesFile(samplesFile);
	mutex_selfGames.unlock();
	return 0;
}


/********************************************** CODINGAME / SUBMIT MODE. IT READS INPUTS FROM CIN, AND IT HAS TIME LIMIT **************************************************************/

int codingame(int argc, char* argv[]) {
#ifdef  DBG_MODE
	cerr << "WARNING!!!!! DEBUG MODE ENABLED!!!!!!!!!!!!" << endl;
#else
	cerr << "CODINGAME" << endl;
#endif
	cacheNNEval.reserve((200 * 6000 + 80000) * 2);
	USE_NNEVAL_CACHE = true;
	//Read command line parameters
	VIEW_TREE_DEPTH = 1;
	stopwatch.Start(400 * 1000);
	int agc = 1;

	string fileModel1 = "best.w32";
	MCTS_Conf conf1 = submit_Mode;
	if (argc > agc) conf1.cpuct_base = (float)atof(argv[agc++]);
	if (argc > agc) conf1.cpuct_inc = (float)atof(argv[agc++]);
	if (argc > agc) conf1.cpuct_limit = (float)atof(argv[agc++]);
	if (argc > agc) conf1.num_iters_per_turn = atoi(argv[agc++]);
	if (argc > agc) conf1.dirichlet_noise_epsilon = (float)atof(argv[agc++]);
	if (argc > agc) conf1.dirichlet_noise_alpha = (float)atof(argv[agc++]);
	if (argc > agc) conf1.dirichlet_decay = (float)atof(argv[agc++]);
	if (argc > agc) conf1.simpleRandomRange = (float)atof(argv[agc++]);
	if (argc > agc) conf1.PROPAGATE_BASE = (float)atof(argv[agc++]);
	if (argc > agc) conf1.PROPAGATE_INC = (float)atof(argv[agc++]);
	if (argc > agc) conf1.POLICY_BACKP_FIRST = atoi(argv[agc++]);
	if (argc > agc) conf1.POLICY_BACKP_LAST = atoi(argv[agc++]);
	//conf1.useHeuristicNN = true;

	MCTS* player = nullptr;
	//nullptr;
	auto gamestate = std::make_shared<_Game>();
	gamestate->readConfig(stopwatch);
	Model model = _Game::CreateNNModel();
	if (!conf1.useHeuristicNN)
	{
		model.loadWeights(fileModel1);
		if (!model.Loaded)
		{
			cerr << "Can't load model " << fileModel1 << endl;
			abort();
		}
		cerr << "using Neural Network " << gamestate->EvalNN(model, 0, 0) << endl;
	}
	vector<Move> bestMove;
	while (true) {
		SIMCOUNT = 0;
		NNCACHE_TOTAL = 0;
		NNCACHE_MISS = 0;
		NNCACHE_HIT = 0;
		gamestate->readTurn(stopwatch);
		cerr << gamestate->WW[0] << " " << gamestate->WW[1] << " ";
		if (player == nullptr)
		{
			player = new MCTS(conf1, &stopwatch, 250 * 1024 * 1024);
		}
		cerr << "Start Search T:" << stopwatch.EllapsedMilliseconds() << "ms" << endl;
		player->Search(model, gamestate, bestMove,&cerr);
		cerr << "End Search Sims:" << SIMCOUNT << " T:" << stopwatch.EllapsedMilliseconds() << "ms";
		cerr << " NNCache:" << NNCACHE_HIT << "/" << NNCACHE_TOTAL << "/" << 100 * (NNCACHE_HIT) / (1 + NNCACHE_TOTAL) << "%";
		cerr << endl;
		cout << PrintMove(bestMove[0], gamestate) << endl;
		//I need to apply my move to calculate my score.
		//Based on my own score I can infer the enemy score at readTurn
		gamestate->Simulate(bestMove);
	}
	return 0;
}



/********* AUXILIARY TOOLS ***************/

//Converting a a file with 32-bit float weights to a 16-bit floats. That means 50% file reduction.
#pragma warning( push )
#pragma warning( disable : 4556 )
void file32to16(string f32, string f16) {
	ifstream I(f32, ios::binary);
	ofstream O(f16, ios::binary);
	if (I.good() && O.good()) {
		ALIGN __m128i H;
		int S = (int)I.tellg();
		I.seekg(0);
		union {
			char B[32];
			__m256 V;
		};
		int e8 = S / 32;
		int r8 = S % 32;
		for (int i = 0; i < e8; ++i)
		{
			I.read(B, 32);
			H = _mm256_cvtps_ph(V, _MM_FROUND_NO_EXC);
			O.write((char*)&H, 16);
		}
		if (r8 > 0)
		{
			I.read(B, r8);
			H = _mm256_cvtps_ph(V, _MM_FROUND_NO_EXC);
			O.write((char*)&H, r8 / 2);
		}
		I.close();
		O.close();
	}
}

//Reverse conversion, 16b to 32b
void file16to32(string f16, string f32) {
	ifstream I(f16, ios::binary);
	ofstream O(f32, ios::binary);
	if (I.good() && O.good()) {
		ALIGN __m256 f;
		int S = (int)I.tellg();
		I.seekg(0);
		union { char B[16]; __m128i H; };
		int e8 = S / 16;
		int r8 = S % 16;
		for (int i = 0; i < e8; ++i)
		{
			I.read(B, 16);
			f = _mm256_cvtph_ps(H);
			O.write((char*)&f, 32);
		}
		if (r8 > 0)
		{
			I.read(B, r8);
			f = _mm256_cvtph_ps(H);
			O.write((char*)&f, r8 * 2);
		}
		I.close();
		O.close();
	}
}
#pragma warning( pop )



void ValidateModel(string file, string str_inputs) {
	Model m = _Game::CreateNNModel(true);
	m.loadWeights(file);
	m.summary();
	Game gm = make_shared<_Game>();
	gm->Reset();


	auto& input = m.inputs[0]->output; //Get input reference
	float* tnn = (float*)&input.xmm[0].v;
	vector<float> values;
	istringstream iss(str_inputs);
	copy(istream_iterator<float>(iss), istream_iterator<float>(), back_inserter(values));
	for (auto& f : values) {
		*tnn = f;
		++tnn;
	}
	cerr << "Inputs:" << str_inputs << endl;
	float val = 0.0f;
	Tensor* policy;

	m.predict();

	policy = &m.outputs[1]->output;
	//Set that value as Score;
	val = m.outputs[0]->output.getElement(0);

	//policy->setElement(6, -99999999.99f);
	//policy->setElement(7, -99999999.99f);
	//Activation_Softmax(*policy, *policy);
	cerr << "VALIDATE MODEL:" << endl;
	cerr << "Value Expected :[" << val << "]" << endl;
	cerr << "Policy expected :[";
	float* tns = (float*)&policy->xmm[0].v;
	for (int i = 0; i < policy->size; ++i) {
		cerr << setprecision(8) << *(tns + i) << " ";
	}
	cerr << "]" << endl;

	cerr << endl;
}


void tmpValidate() {
	/*file32to16("candidate.weights", "candidate.w16");
	file16to32("candidate.w16", "candidate.w32");
	file32to16("candidate.w32", "check.w16");
	file16to32("check.w16", "check.w32");*/

	ValidateModel("gen0000.w32",
		R"(1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 1. 0. 0. 0. 0. 0.)");
}
int main(int argc, char* argv[]) {
//Paste here your Model validation. Once validated, remove these lines.

	string PROGRAMNAME(argv[0]);
	NNCACHE_TOTAL=0;
	NNCACHE_MISS=0;
	NNCACHE_HIT=0;
	if (argc > 1)
	{
		string tmpOption(argv[1]);

#ifdef DBG_MODE
		if (tmpOption == "test")
		{
			tmpValidate();
			return 0;
		}
#endif
		if (tmpOption == "selfplay")
		{
			return selfPlay(argc, argv);
		}
		if (tmpOption == "pitplay")
		{
			return pitPlay(argc, argv);
		}
	}
	return codingame(argc, argv);
}
