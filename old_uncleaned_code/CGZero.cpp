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
//https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a
//https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191
#define CFG_Game_IsSavedInMCTSNode 1
//#define OPENING_BOOK
#ifdef OPENING_BOOK
#include <unordered_map>
std::unordered_map<uint64_t, float> opening_book;
uint64_t BOOKHIT = 0;
uint64_t BOOKTOTALS = 0;
#endif

#include "NN_Mokka.h"

using namespace std;

const int MAX_TURNS = 199;
const int ROLLOUT_DEPTH = 0;

//Use softmax for policy selfPlay
bool REPLAY_SOFTMAX_POLICY = false;

#define REMOVE_SOFTMAX_INVALID_MOVES
int VIEW_TREE_DEPTH = 2;


enum mcts_mode { selfplay, pit, submit };
struct MCTS_Conf {
	float cpuct_base = 0.02f;
	float cpuct_inc = 4.98f / 200.0f;
	float cpuct_limit = 5.0f;

	float dirichlet_noise_epsilon = 0.25f;
	float dirichlet_noise_alpha = 1.0f;	// Dirichlet alpha = 10 / n --> Max expected moves
	float dirichlet_decay = 0.02f;
	int num_iters_per_turn = 900;
	float simpleRandomRange = 0.00f;
	bool useTimer = false;
	bool useHeuristicNN = false;
	float PROPAGATE_BASE = 0.7f; //Propagate "WIN/LOSS" with 30% at start
	float PROPAGATE_INC = 0.3f; //Linearly increase +70% until endgame
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

vector<string> DEBUG_STR = {

};

unordered_map<string, int> DEBUG_FORCESELECT = {


};

unordered_map < string, vector<double>> DEBUG_POL = {
};


MCTS_Conf selfPlay_Mode(0.02f, 0.03f, 2.0f, 0.25f, 1.6f, 0.01f, false, 1200, 0.00f, 0.7f, 0.3f, 10, 10, mcts_mode::selfplay);
MCTS_Conf pit_Mode(0.02f, 0.03f, 2.0f, 0.05f, 1.6f, 0.007f, false, 1200, 0.00f, 0.7f, 0.3f, 10, 10, mcts_mode::pit);
MCTS_Conf submit_Mode(3.0f, 0.0f, 3.0f, 0.00f, 1.3f, 0.0000f, true, 100, 0.05f, 0.7f, 0.3f, 10, 10, mcts_mode::submit);
//MCTS_Conf submit_Mode(3.0f, 0.0f, 3.0f, 0.03f, 1.3f, 0.0007f, true, 100, 0.02f, 0.7f, 0.3f, 10, 10, mcts_mode::submit);

#ifdef _MSC_VER
#define TIMEOUT_0 900 * 1000 *50
#define TIMEOUT_N 90 * 1000 *50
#define ALIGN __declspec(align(32))
#else 
#define TIMEOUT_0 860 * 1000
#define TIMEOUT_N 139000
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
	/*	float invertScore(const int& playSize, const float& originalScore, const int& newDepth, const uint8_t* savedGame);
		int getID_at_Turn(const int& _depth, const uint8_t* savedGame);*/

	float getInitialTemperature() {
		if (turn < 30 || turn == 255)  //More randomness the first 5 turns
			return 1.0f;
		else return 0.0f;
	}

	bool isEndGame();
	int getWinnerID();
	int nextPlayer();
	int getIDToPlay();
	int getTimeLimit();
	//void GenerateAllPossibleMoves() { DBG(cerr<<"GENALL"<<endl;);abort();}
	int getPlayerCount();
	//To export policies
	void swapPlayers();
	uint64_t CalcHash(const int& _NextMove)noexcept;

	//Simultaneous moves. When the game is sequential there will only be 1 concurrent unit.
	//Possible modes can vary with depth. I.e. some games only have expensive calculated moves at firsts turns
	void getPossibleMoves(const int& concurrentUnitID, vector<Move>& Possible_Moves, const int& _depth);
	void Simulate(const vector<Move>& concurrentUnitMoves);
	void Simulate(const Move& singleMove);
	//Get players and units.

	bool Equals(const Game& g);
	bool Equals(const _Game& g);
	//Evaluation
	float EvalPlayer(const int& playerID, const int& _depth);
	bool EvalBook(float& Score, const int& playerID);
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
			uint8_t gameEnded : 1;//7 bits available
			uint8_t idToPlay : 1;//6 bits available
			uint8_t lastMove : 3;//3 bits available
			uint8_t swapped : 1;//0 bits available
			uint8_t freeBits : 2;//0 bits available
			uint8_t score1;
			uint8_t cell1[6];
		};
	};

	string FIRMA()
	{
		string s = "";
		for (int i = 0; i < 6; ++i)
		{
			s += to_string((int)cell0[i]);
		}
		s += " " + to_string((int)score0) + "|";
		for (int i = 0; i < 6; ++i)
		{
			s += to_string((int)cell1[i]);
		}
		s += " " + to_string((int)score1) + "|";
		return s;
	}


};

const uint8_t IS_LEAF = 0; //Simpler to reset
const uint8_t NO_CHILDREN = 255;


inline int _Game::getPackSize()noexcept { return _Game::PACKED_SIZE; }
inline int _Game::getIDToPlay() { return idToPlay; }
inline int _Game::nextPlayer() { return 1 - idToPlay; }

inline void _Game::CopyFrom(const Game& original) {
	WW[0] = original->WW[0];
	WW[1] = original->WW[1];
}
inline void _Game::CopyFrom(const _Game& original) {
	WW[0] = original.WW[0];
	WW[1] = original.WW[1];
}


int _Game::getTimeLimit() {
	if (turn == 0) return 550 * 1000; else  return 41 * 1000;
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
	CalcHash(-1);
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
	CalcHash(-1);
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
	//int idToPlay = getIDToPlay();
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

#ifdef DBG_MODE
	if ((int)Possible_Moves.size() > 6) {
		cerr << "Fallo " << (int)Possible_Moves.size() << " > " << 6 << endl;
	}
#endif

	//Special case
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


uint64_t _Game::CalcHash(const int& _NextMove)noexcept {
	uint64_t Hash = 0ULL;
	const uint64_t MASK_CELLA = 0x0E010E010E010000ULL; //First bit of 6 cells
	const uint64_t MASK_CELLB = 0x010E010E010E0000ULL; //remaining bits 3*6 = 18

	Hash = _pext_u64(WW[0], MASK_CELLA)  //0  - 12 bits from cell0
		+ (_pext_u64(WW[1], MASK_CELLA) << 12) // 12 bits from cell1
		+ (_pext_u64(WW[1], MASK_CELLB) << 24) // 12 bits from cell1
		+ (_pext_u64(WW[1], MASK_CELLB) << 36) // 12 bits from cell1
		;
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
	/*cerr << Print() << endl;
	cerr << bitset<64>(Hash)<<" : "<<Hash << endl;*/
	return Hash;
}

void _Game::Pack(uint8_t* g)noexcept {
	uint64_t* target = (uint64_t*)g;
	target[0] = WW[0];
	target[1] = WW[1];
#ifdef _MSC_VER
#ifdef DBG_MODE
	//Validacion
	_Game PRU;
	PRU.Unpack(g);
	if (!PRU.Equals(*this))
	{
		cerr << "ERROR EMPAQUETADO" << endl;
	}
#endif
#endif
}
void _Game::Unpack(uint8_t* g)noexcept {
	uint64_t* src = (uint64_t*)g;
	WW[0] = src[0];
	WW[1] = src[1];
}

_Game::_Game() {
	//GenerateAllPossibleMoves();
}
_Game::_Game(const Game& original) {
	CopyFrom(original);
}
_Game::_Game(const _Game& original) {
	CopyFrom(original);
}

int  _Game::getInputDimensions() { return 6 * 2 * 24 + 2 * 27; }
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
	policy = (*NN(Dense)("Policy", POLICY_SIZE, (activeSoftMax ? SOFTMAX : NONE)))(p1); //it should be softmax, but we are normalizing after dirichlet and move restrictions
	//policy = (*NN(Dense)("Policy", POLICY_SIZE, SOFTMAX))(x); //it should be softmax, but we are normalizing after dirichlet and move restrictions

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


//Definition of MCTS based on the real game



//Training related info
struct ReplayMove {
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

//SELF-PLAY: SAMPLES
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
/************************** MCTS **********************************/
Random SMIT_rnd = Random();
struct MCTS_Node;
struct PRECACHE_PACK;
struct PRECACHE;


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


const uint16_t PRECACHE_NODECOUNT_PACK = 65535;
struct NodeIndex {
	uint16_t BlockID; //This allows tree reuse without exhausting nodes or collisions
	uint16_t FirstChild;
};

const uint8_t Status_GAMESTATE_SAVED = 1 << 0;

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
	maxScore = _policy;

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
class MCTS {
public:
	MCTS_Conf conf;
	PRECACHE cache;
	Stopwatch* timecontrol = nullptr;
	Game lastTurn = nullptr;
	Random rnd;
	vector<Move> MKCHlist;
	int rolloutCount = 0;

	//Outer vectors have size == 1 on Sequential games
	vector<MCTS_Node*> rootNodes;
	vector<MCTS_Node*> bestNodeToPlay;
	//vector<vector<MCTS_Node*>> playedNodes;
#ifdef DBG_MODE
	unordered_set< MCTS_Node*> tmpChecksingleUse;
#endif
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
		*DEBUG_VIEW_TREE << ID << ":D" << (int)N->depth << " N:" << N->visits << "|" << sumChildVisits << " Vn:" << N->nnValue << " Sum:" << N->sumScore << "|" << sumValues << " Mean:" << N->sumScore / (float)max(1, N->visits) << " Pol:" << N->policy << " Child:" << (int)N->ChildCount << " St:" << (int)N->Status << endl;
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
	ReplayMove getReplayBuffer(Model& model, Game& gamestate) {
		ReplayMove rm;

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
		//Invalids are set as negative
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


#ifdef DBG_MODE
		float tmpConfirmNormalize = 0.0f;
		for (auto& ff : rm.policy)
		{
			if (ff >= 0.0f)
				tmpConfirmNormalize += ff;
		}
		if (rm.validMovesNr > 0)
			if (tmpConfirmNormalize < 0.98f || tmpConfirmNormalize > 1.02f)
			{
				auto node = cache.getNode(rootNodes[0]->ChildIndex);
				float tmpHJ = 0.0f;
				for (int i = 0; i < rm.validMovesNr; ++i) {
					tmpHJ += (node + i)->policy;
					cerr << i << ":" << (node + i)->policy << " " << tmpHJ << endl;
				}
				cerr << "Error normalize on replay" << endl;
			}
#endif
		rm.selectedMove = bestNodeToPlay[0]->action;
		if (rm.validMovesNr == 0)
			rm.selectedMove = 0;

		if (rm.currIDToPlay != 0) {
			gamestate->swapPlayers();
		}

		return rm;
	}

	void pickBestPlay(const Game& gamestate, vector<Move>& bestMove, ostream* DEBUG_VIEW_TREE = nullptr) {
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
		//Not sure about this	apply_temperature(gamestate);

		int Visits = 0;

		int tmpCheckVisits = 0;


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

			int nCount = rootNode->ChildCount;

			for (int i = 0; i < nCount; ++i) {
				Visits += node->visits;
				if (DEBUG_VIEW_TREE)
				{
					*DEBUG_VIEW_TREE << node->visits << " " << node->policy << " -> " << ((float)node->visits) * sumVisits;
				}
				if (node->visits > mostVisited->visits)
				{
					mostVisited = node;
					if (DEBUG_VIEW_TREE)
					{
						*DEBUG_VIEW_TREE << " *";
					}
				}
				if (DEBUG_VIEW_TREE)
				{
					_Game tGM;
					tGM.Unpack(node->savedGameState);
					*DEBUG_VIEW_TREE << " SC:" << (int)tGM.score0 << " " << (int)tGM.score1;
					*DEBUG_VIEW_TREE << endl;
				}
				node++;
			}
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
			if (score > node->maxScore)
				node->maxScore = score;
			++node->visits;
			node = node->parent;
			score = -score;
		}
#ifdef DBG_MODE
		node = leaf;
		int tmpDepth = 0;
		while (node != nullptr)
		{
			tmpValidateVisitConsistency(node);
			tmpParentCheck(node);
			node = node->parent;
			++tmpDepth;
		}
#endif
	}


	inline MCTS_Node* Select(MCTS_Node* parent, MCTS_Node* firstChild, int _depth, int _turn) {
		if (parent->ChildCount == 1)
		{
			return firstChild;
		}
		MCTS_Node* child = firstChild;//cache.getNode(CacheIndex);
		//float turn_cpuct = max(conf.cpuct_base, min(conf.cpuct_limit, conf.cpuct_base + conf.cpuct_inc * (float)_turn));
		float turn_cpuct;
		if (conf.cpuct_inc > 0.0f)
		{
			turn_cpuct = min(conf.cpuct_limit, conf.cpuct_base + conf.cpuct_inc * (float)_turn);
		}
		else {
			turn_cpuct = max(conf.cpuct_limit, conf.cpuct_base + conf.cpuct_inc * (float)_turn);
		}
		float parent_F = (parent->visits <= 1 ? turn_cpuct : turn_cpuct * fastsqrtf((float)parent->visits));

		/*

		*/
		MCTS_Node* bestChild = child;
		float bestUCT = -9999999.0f;
		for (int i = 0; i < parent->ChildCount; ++i)
		{
			float Q = (child->visits <= 1 ? child->sumScore : child->sumScore * fastinv(child->visits));
			//float P = iszero(ϵ) ? policy : (1-ϵ) * policy + ϵ * η[i]
			float U = parent_F * child->policy * fastinv(1 + child->visits);

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
		float turn_cpuct = min(conf.cpuct_limit, conf.cpuct_base + conf.cpuct_inc * (float)_turn);
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
#ifdef DBG_MODE
		float tmpConfirmNormalize = 0.0f;
#endif
		for (int i = 0; i < parent->ChildCount; i++)
		{
			child->policy = child->policy * (1.0f - epsilon) + dirichlet_vector[i] * factorDirich;
#ifdef DBG_MODE
			tmpConfirmNormalize += child->policy;
#endif
			++child;
		}
#ifdef DBG_MODE
		if (tmpConfirmNormalize < 0.95f || tmpConfirmNormalize > 1.05f)
		{
			cerr << "Error normalize dirichlet " << tmpConfirmNormalize << endl;
		}
#endif
	}

	inline void tmpParentCheck(MCTS_Node* parent)
	{
#ifdef DBG_MODE
		for (auto& tt : parent->tmpChildren)
		{
			if (tt->parent != parent)
			{
				cerr << "Parent error:" << tt->parent << " != " << parent << endl;
				cerr << "ERORROR" << endl;

			}
		}
#endif
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
				parent->nnValue = working->EvalPlayer(1 - ownerID, _depth);
			}
			if (parent->parent == nullptr)
			{
				parent->nnValue = -parent->nnValue;
			}
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
				//*********
				for (int i = 0; i < childCount; ++i)
				{
					(child + i)->Reset(parent, MKCHlist[i], 0.0f);
					float child_pol;

					if (!conf.useHeuristicNN)
					{
						child_pol = policy->getElement((uint32_t)MKCHlist[i]);
					}
					else {
						tmpGScalc.CopyFrom(working);
						tmpGScalc.Simulate(MKCHlist[i]);
						tmpGScalc.Pack(&(child + i)->savedGameState[0]);
						(child + i)->isGameStateSaved = 1;
						child_pol = tmpGScalc.EvalPlayer(ownerID, _depth + 1);
					}
					//		bckPolicy2[MKCHlist[i]] = child_pol;
					if (conf.simpleRandomRange > 0.0f)
					{
						float randNoise = 1.0f + rnd.NextFloat(-conf.simpleRandomRange, conf.simpleRandomRange);
						child_pol *= randNoise;
					}

					(child + i)->policy = child_pol;
					sumPolicy += child_pol;
				}
				if (conf.useHeuristicNN)
				{ //Policy normalization
					string HS = to_string(working->WW[0]) + " " + to_string(working->WW[1]);
					if (DEBUG_POL.find(HS) != DEBUG_POL.end()) {
						auto& N = DEBUG_POL[HS];
						for (int i = 0; i < childCount; ++i) {
							(child + i)->policy = (float)N[i];
						}
					}
					else {

#ifdef DBG_MODE
						sumPolicy = 0.0f;
#endif
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
						float* tns = &policy->xmm[0].f[0];
						for (int i = 0; i < childCount; ++i) {
							*(tns + (child + i)->action) = (child + i)->policy;
						}
						Activation_Softmax(*policy, *policy);
						for (int i = 0; i < childCount; ++i) {
							(child + i)->policy = *(tns + (child + i)->action);
#ifdef DBG_MODE
							sumPolicy += (child + i)->policy;
#endif

						}
					}
				}

				//Policy Normalization to valid moves
#ifdef DBG_MODE
				//if (sumPolicy > 0.05f)
				{
					/*		if (sumPolicy <= 0.955f)
							{
								cerr << "Policy Error "<< sumPolicy << endl;
								MCTS_Node* child = cache.getNode(parent->ChildIndex);
								for (int i = 0; i < policy->size; ++i)
								{
									cerr<<"Element:" << i << ":NN:" << bckPolicy[i]<<"  SOFTMAX:"<< " "<< bckPolicy2[i];
									for (int j = 0; j < childCount; ++j)
									{
										if ((uint32_t)MKCHlist[j] == i)
										{
											cerr << " * OnMCTS:"<<(child+j)->policy;
										}
									}
									cerr << endl;
								}
								cerr << endl;
							}*/
							/*
							sumPolicy = 1.0f / sumPolicy;
							float tmpConfirmNormalize = 0.0f;
							for (int i = 0; i < childCount; ++i)
							{
								(child + i)->policy *= sumPolicy;
								tmpConfirmNormalize += (child + i)->policy;
							}
							if (tmpConfirmNormalize < 0.997f || tmpConfirmNormalize > 1.003f)
							{
								cerr << "Error normalize " << tmpConfirmNormalize << endl;
								for (int i = 0; i < childCount; ++i)
								{
									auto child_pol = policy->getElement((uint32_t)MKCHlist[i]);
									if (conf.simpleRandomRange > 0.0f)
									{
										float randNoise = 1.0f + rnd.NextFloat(-conf.simpleRandomRange, conf.simpleRandomRange);
										child_pol *= randNoise;
									}
									(child + i)->Reset(parent, MKCHlist[i], child_pol);
									sumPolicy += child_pol;
								}
							}*/
				}
#endif
				if (parent->parent == nullptr && _depth == 0 && conf.dirichlet_noise_epsilon > 0.0f)
				{
					dirichlet_noise(parent, conf.dirichlet_noise_epsilon - conf.dirichlet_decay * (float)working->turn, conf.dirichlet_noise_alpha);
				}
			}
#ifdef DBG_MODE
			for (int i = 0; i < childCount; ++i) {
				parent->tmpChildren.push_back(child + i);
				if (tmpChecksingleUse.find(child + i) != tmpChecksingleUse.end())
				{

					cerr << "Error, children already in use!!!!" << (child + i)->printGraph(cache);
					cerr << "ERORROR" << endl;
				}
				tmpChecksingleUse.emplace(child + i);
			}
			tmpParentCheck(parent);
#endif
		}
		return childCount;
	}

	void RestoreRoot(const Game& gamestate) {
		//Tree Reuse - Recover new root from current tree
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



	inline void tmpValidateVisitConsistency(MCTS_Node* current) {
#ifdef DBG_MODE
		if (current->ChildCount == 0 || current->ChildCount == NO_CHILDREN)
			return;
		auto node = cache.getNode(current->ChildIndex);
		int parentVisits = 0;
		for (int i = 0; i < current->ChildCount; ++i)
		{
			parentVisits += (node + i)->visits;

		}
		if (abs(current->visits - parentVisits) > 1)
		{
			cerr << "Not consistent: child:" << parentVisits << "!= Sum:" << current->visits << endl;
			parentVisits = 0;
			for (int i = 0; i < current->ChildCount; ++i)
			{
				parentVisits += (node + i)->visits;
				cerr << (node + i)->visits << " " << parentVisits << endl;
			}

			cerr << "ERROR " << current->printGraph(cache) << endl;
			cerr << "-----------------------------------------" << endl;
		}
#endif
	}
	int maxDepth = 0;
	//Classic MCTS - Turn based games, 2 players
	int Search(Model& model, const Game& gamestate, vector<Move>& bestMove, ostream* DEBUG_VIEW_TREE = nullptr) {
#ifdef DBG_MODE
		tmpChecksingleUse.clear();
#endif
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

		//		int CURR_DEPTH = 210;// 8; //VARIABLE DEPTH.
		while ((!conf.useTimer && rolloutCount < conf.num_iters_per_turn) || (conf.useTimer && !timecontrol->Timeout()))
		{
			int _depth = 0;
			current = rootNodes[0];
			working->CopyFrom(gamestate);
			//Tree traverse until leaf, no simulation 
			while (current->ChildCount != IS_LEAF && current->ChildCount != NO_CHILDREN && !working->isEndGame())
			{
				current = Select(current, cache.getNode(current->ChildIndex), _depth, working->turn);

				tmpValidateVisitConsistency(current);
				tmpParentCheck(current);
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
				tmpValidateVisitConsistency(current);
				tmpParentCheck(current);
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
				tmpParentCheck(current);
				++_depth;

			}
			tmpValidateVisitConsistency(current);
			tmpParentCheck(current);
			backPropagate(current);
			tmpValidateVisitConsistency(current);
			maxDepth = max(maxDepth, 1 + _depth);

			++rolloutCount;
		}
		DBG(if (conf.mode == mcts_mode::submit) cerr << "End Search, Max Depth:" << maxDepth << " Rollout Count:" << rolloutCount << endl;);
		pickBestPlay(gamestate, bestMove, DEBUG_VIEW_TREE);

		//cerr<<"Search visits:" << rootNodes[0]->visits << endl;
		working->CopyFrom(gamestate);

		return 0;
	}


};


Model NNplayer[2];
//For self-play we load 2 models
Model LoadOwareModel(const Game game, int playerID, string filename)
{
	Model m = game->CreateNNModel();
	m.loadWeights(filename);
	return m;
}


void _Game::setNNInputs(Model& model, const int& playerID) {
	//Neural Network inputs are always mirrored as player 1.
	auto& input = model.inputs[0]->output; //Get input reference
	//clear;
	for (int i = 0; i < input.xmm_size; ++i)
	{
		input.xmm[i].v = _mm256_setzero_ps();
	}
	int off0 = (playerID == 0 ? 0 : 6 * 24);
	int off1 = (playerID == 0 ? 6 * 24 : 0);


	float* tns = (float*)&input.xmm[0].v;

	//One-hot encoding, it seems to work better
	for (int i = 0; i < 6; ++i)
	{
		*(tns + off0 + 24 * i + (cell0[i] > 24 ? 24 : cell0[i])) = 1.0f;
		*(tns + off1 + 24 * i + (cell1[i] > 24 ? 24 : cell1[i])) = 1.0f;
	}
	off0 = 12 * 24 + (playerID == 0 ? 0 : 1 * 27);
	off1 = 12 * 24 + (playerID == 0 ? 1 * 27 : 0);
	*(tns + off0 + (score0 > 27 ? 27 : score0)) = 1.0f;
	*(tns + off1 + (score1 > 27 ? 27 : score1)) = 1.0f;
}

void _Game::predict(Model& model, const int& playerID, Tensor** policy, float& nVal)
{
	//Todo, hash caching of predictions
	setNNInputs(model, playerID);
	model.predict();
	*policy = &model.outputs[1]->output;
	//Set that value as Score;
	nVal = -model.outputs[0]->output.getElement(0);
	//TODO: This is true!?!?!?!?!?!?
	//if (idToPlay == 0)
		//nVal = -nVal;
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
//Removed, it was just used for testing of the sample output.
	abort();
	return Score;
}

bool _Game::EvalBook(float& Score, const int& playerID) {
#ifdef OPENING_BOOK
	++BOOKTOTALS;
	uint64_t Hash = CalcHash(-1);
	auto busca = opening_book.find(Hash);
	if (busca == opening_book.end())
	{
		_Game g;
		g.CopyFrom(*this);
		g.SetAsPlayer0();
		Hash = CalcHash(-1);
		busca = opening_book.find(Hash);
		if (busca == opening_book.end())
		{
			return false;
		}
		else Score = -busca->second;
		++BOOKHIT;
		return true;
	}
	++BOOKHIT;
	Score = busca->second;
	return true;
#else
	return false;
#endif
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
#ifdef OPENING_BOOK
	else if (EvalBook(Score, playerID)) {
	}
#endif
	//else if (NNevaluator != nullptr) { Score = EvalNN(playerID, _depth) * 0.2f + 0.8f * EvalHeuristic(playerID, _depth); }
	else {
		Score = EvalHeuristic(playerID, _depth);
		Score *= sc.NextFloat(0.95f, 1.05f);
	}


	return (playerID == 0 ? Score : -Score);
}


atomic<int> Pit_V1;
atomic<int> Pit_V2;
atomic<int> Pit_Draw;
atomic<int> matches;
void Worker_Pit(int ID, string fileModel1, string fileModel2, int matchperWorker, MCTS_Conf conf1, MCTS_Conf conf2)
{
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
			cerr << "Worker " << ID << ": " << Pit_V1 << "/" << Pit_V2 << "/" << Pit_Draw << ":" << winrate << "%" << endl;
		}

		++matches;
	}
}

int pitPlay(int argc, char* argv[])
{
	Pit_V1 = 0;
	Pit_V2 = 0;
	Pit_Draw = 0;
	matches = 0;
	int agc = 2;
	int THREADS = atoi(argv[agc++]);
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


ReplayBuffer selfGames;
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
		cerr << "Worker " << ID << ":" << matches << " W:" << RG.reward << " " << (int)ws->score0 << " " << (int)ws->score1 << " T:" << (int)ws->turn << "/" << RG.moves.size() << " Sw:" << (int)ws->swapped << " MAX DEPTH:p1:" << p0->maxDepth << "/p2:" << p1->maxDepth << endl;
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


void saveBufferReplays(string fileModel, bool binaryMode) {
	//Save to a file
	auto t = time(nullptr);
	auto tm = *localtime(&t);
	ostringstream oss;
#ifdef _MSC_VER
	oss << "./traindata3/Replay_" << put_time(&tm, "%Y%m%d_%H%M%S") << "_" << fileModel;
#else
	oss << "./traindata/Replay_" << put_time(&tm, "%Y%m%d_%H%M%S") << "_" << fileModel;
#endif
	string filename = oss.str();
	ofstream f(filename, (binaryMode ? (ios::out | ios::binary) : ios::out));
	for (auto& G : selfGames.games)
	{
		for (auto& h : G.moves)
		{
			if (binaryMode)
			{
				assert((int)h.gamestate.size() == _Game::getInputDimensions());
				assert((int)h.policy.size() == 6);
				f.write(reinterpret_cast<const char*>(&h.gamestate[0]), (int)h.gamestate.size() * sizeof(float));
				f.write(reinterpret_cast<const char*>(&h.policy[0]), (int)h.policy.size() * sizeof(float));
				f.write(reinterpret_cast<const char*>(&h.meanScore), sizeof(float));
			}
			else
			{
				for (auto& i : h.gamestate) {
					f << i << " ";
				}
				for (auto& p : h.policy) {
					f << p << " ";
				}
				f << h.meanScore << endl;
			}
		}
	}
	f.close();
}
void debugSaveBufferReplays(string fileModel) {
	//Save to a file
	auto t = time(nullptr);
	auto tm = *localtime(&t);
	ostringstream oss;
	oss << "_" << put_time(&tm, "%Y%m%d_%H%M%S") << "_" << fileModel;
	string filename = "./traindata/Debug"+oss.str();
#ifdef _MSC_VER
	string fileMoves = "./traindata3/Moves"+oss.str();
	filename = "./traindata3/Debug" + oss.str();
	ofstream fMoves(fileMoves);
#endif
	
	ofstream f(filename);
	for (auto& G : selfGames.games)
	{
		f << "NEW GAME    **********************************************************************************" << endl;
#ifdef _MSC_VER
		fMoves << "NEW GAME    **********************************************************************************" << endl;
#endif
		int turn = 0;
		for (auto& h : G.moves)
		{
#ifdef _MSC_VER
			fMoves << h.selectedMove << endl;
#endif
			f << "T:" << turn << "    **********************************************************************************" << endl;
			int seedCount = 0;
			f << " GS:";
			int index = 0;

			for (int p = 0; p < 2; ++p)
			{
				f << "P" << p << ":";
				for (int i = 0; i < 6; ++i)
				{
					for (int v = 0; v < 24; ++v)
					{
						if (h.gamestate[index++] > 0.9f)
						{
							seedCount += v;
							f << v << ",";
						}
					}

				}
				f << "|";
			}
			for (int p = 0; p < 2; ++p)
			{
				f << "SC" << p << ":";
				for (int v = 0; v < 27; ++v)
				{
					if (h.gamestate[index] > 0.9f)
					{
						f << v << " ";
						seedCount += v;
					}
					++index;
				}
				f << "|";
			}
			f << " Seeds:" << seedCount << " ";
			f << " currIdToPlay:" << h.currIDToPlay;
			f << " tmpSwapped:" << h.tmpSwapped;
			f << " tmpIDToPlay:" << h.tmpIDToPlay;
			f << " VALUE:" << h.meanScore;
			//TEMP
			f << " ORIG " << h.originalValue << " BACK " << h.backtrackValue;

			/*		for (auto& i : h.gamestate) {
						f << i << " ";
					}*/
			f << "POLICY:[";
			float tmpSUMPOL = 0.0f;
			for (auto& p : h.policy) {
				f << p << ",";
				tmpSUMPOL += p;
			}
			f << "] Sum:" << tmpSUMPOL;



			//TEMP
			f << " factorBPP:" << h.factorBPP << " ORIPOL:[";
			float tmpORIPOL = 0.0f;
			for (auto& nn : h.originalpolicy)
			{
				f << nn << ",";
				tmpORIPOL += nn;
			}
			f << "] Sum:" << tmpORIPOL;

			f << " SC:" << (int)h.testGame.score0 << " " << (int)h.testGame.score1;
			f << " " << h.testGame.WW[0] << " " << h.testGame.WW[1];
			f << endl;
#ifdef _MSC_VER
			f << h.SearchResult << endl;
#endif
			++turn;
		}
		auto& B = G.moves.back();
		if (abs(B.originalValue - B.backtrackValue) > 0.3f)
		{
			cerr << "Desvio: " << B.originalValue << " | " << B.backtrackValue << endl;
		}
	}
	f.close();
#ifdef _MSC_VER
	fMoves.close();
#endif
}
/**Self-play for training **/
int selfPlay(int argc, char* argv[])
{
	matches = 0;
	int agc = 2;
	int THREADS = atoi(argv[agc++]);
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

	//Save to a file
	string samplesFile = "./traindata/Replay_" + fileModel1 + "_" + fileModel2 + ".dat";
#ifdef _MSC_VER
	samplesFile = "./traindata3/Replay_" + fileModel1 + "_" + fileModel2 + ".dat";
#endif
	processSamplesFile(samplesFile, _Game::getInputDimensions(), _Game::getPolicySize() + 1); //policy + value as output
	cerr << "selfplay m1:" << fileModel1 << " m2:" << fileModel2 << " Th:" << THREADS << " N:" << matchCount << " C1:" << conf1.print() << " C2:" << conf2.print() << endl;
	selfGames.games.resize(0);
	selfGames.games.reserve(matchCount);
	vector<thread> threads(max(1, THREADS));
	for (int i = 0; i < max(1, THREADS); i++)
	{
		threads[i] = thread(Worker_SelfPlay, i, fileModel1 + ".w32", fileModel2 + ".w32", matchperWorker, conf1, conf2);
	}

	for (int i = 0; i < max(1, THREADS); i++)
	{
		threads[i].join();
	}

	mutex_selfGames.lock();
	SamplesFile* sFile = getSampleFile(samplesFile);
	for (auto&G : selfGames.games)
	{
		for (auto& R : G.moves)
		{
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
			insertNewSample(sFile, S);
		}
	}
	saveSamplesFile(samplesFile);
	mutex_selfGames.unlock();

	//saveBufferReplays(fileModel2 + "vs" + fileModel1 + ".dat", true);
#ifdef _MSC_VER
	//saveBufferReplays(fileModel2 + "vs" + fileModel1 + ".txt", false);
	debugSaveBufferReplays(fileModel2 + "vs" + fileModel1 + ".txt");
#endif
	return 0;
}


int simTest(int argc, char* argv[])
{
	MCTS* player1 = new MCTS(&stopwatch, 350 * 1024 * 1024);
	MCTS* player2 = new MCTS(&stopwatch, 350 * 1024 * 1024);
	auto gs_player1 = std::make_shared<_Game>();
	gs_player1->readConfig(stopwatch);
	auto gs_player2 = std::make_shared<_Game>();
	gs_player2->readConfig(stopwatch);

	vector<Move> bestMove;
	vector<Move> pm;

	float valueORI = 1.0;

	Model NNplayer;
	NNplayer = gs_player1->CreateNNModel();
	NNplayer.loadWeights("best.model");
	NNplayer.saveWeights("best.test");
	NNplayer.summary();
	float NNeval = gs_player1->EvalPlayer(0, 0);
	cerr << "NN gives " << NNeval << " points" << endl;

	vector<float> TryValues = { 1.0f,1.3f,1.5f,1.8f,2.1f,2.6f,3.0f,3.4f, 0.5f,0.75f,0.9f };
	vector<float> winrate;
	vector<int> wins;
	vector<int> games;
	wins.resize(TryValues.size());
	winrate.resize(TryValues.size());
	games.resize(TryValues.size());

	for (int t = 0; t < TryValues.size(); ++t)
	{
		float valueTEST = TryValues[t];
		//for (int position = 0; position < 2; ++position)
		{
			for (int TIMES = 0; TIMES < 250; ++TIMES)
			{
				int position = TIMES & 1;
				stopwatch.Start(400 * 1000);
				gs_player1->Reset();
				for (auto& r : player1->cache.cache_Node)
				{
					r->clear();
				}
				player1->bestNodeToPlay.clear();
				player1->lastTurn = nullptr;
				gs_player1->turn = 0;
				vector<Move> tmpMoves;
				while (true) {
					DBG(cerr << "******************************************************************************************************" << endl;);
					assert(gs_player1->idToPlay == 0);
					stopwatch.Start(gs_player1->getTimeLimit());

					//				CONSTANT_C = position == 0 ? valueTEST : valueORI;
					int result = player1->Search(NNplayer, gs_player1, bestMove);
					gs_player1->Simulate(bestMove);
					gs_player1->getPossibleMoves(0, tmpMoves, 0);
					if (gs_player1->isEndGame())
						break;
					gs_player2->CopyFrom(gs_player1);
					gs_player2->swapPlayers();
					//			CONSTANT_C = position != 0 ? valueTEST : valueORI;
					player2->Search(NNplayer, gs_player2, bestMove);
					gs_player1->Simulate(bestMove);
					gs_player1->getPossibleMoves(0, tmpMoves, 0);
					if (gs_player1->isEndGame())
						break;
				}
				if ((position == 0 && gs_player1->score0 >= gs_player1->score1) ||
					(position == 1 && gs_player1->score1 >= gs_player1->score0))
				{
					++wins[t];
				}
				++games[t];
				cerr << "Finished game. Turn:" << (int)gs_player1->turn << " Position:" << position << " Scores:" << (int)gs_player1->score0 << "/" << (int)gs_player1->score1;
				winrate[t] = (float)wins[t] / (float)games[t] * 100.0f;
				cerr << " WINRATE: C=" << valueTEST << " = " << winrate[t] << "%";
				cerr << endl;
			}
		}
		winrate[t] = (float)wins[t] / (float)games[t] * 100.0f;
		cerr << "WINRATE: C=" << valueTEST << " = " << winrate[t] << "%" << endl;
	}

	for (int t = 0; t < TryValues.size(); ++t)
		cerr << "[" << TryValues[t] << "," << winrate[t] << "%]," << endl;

	return 0;
}



#ifdef OPENING_BOOK
#ifdef _MSC_VER
void createOpeningBookFrom(string s, string opening) {
	float quantize[257];
	for (int i = 0; i < 257; ++i)
	{
		quantize[i] = ((float)i) / 127.0f - 1.0f;
	}
	std::ifstream input(s);
	if (input.good())
	{
		std::ofstream saveToFile(opening, std::ios::binary);


		_Game reader;
		reader.Reset();
		while (!input.eof())
		{
			int val[17];
			for (int i = 0; i < 17; ++i)
			{
				input >> val[i];
			}
			float sumScore;
			input >> sumScore; input.ignore();
			for (int i = 0; i < 6; ++i)
			{
				reader.cell0[i] = val[i];
				reader.cell1[i] = val[6 + i];
			}
			reader.score0 = val[12];
			reader.score1 = val[13];
			int visitas = val[16];
			uint8_t quant = 0;
			float score = sumScore / (float)visitas;
			if (score <= -0.995f)
				quant = 0;
			else if (score >= 0.995f)
				quant = 255;
			else {
				for (int i = 0; i < 255; ++i)
				{
					if (quantize[i] < score && score < quantize[i + 1])
					{
						quant = (uint8_t)i;
					}
				}
			}
			uint64_t Hash = reader.CalcHash(-1);
			//Hash + quant
			saveToFile.write((char*)&Hash, sizeof(Hash));
			saveToFile.write((char*)&quant, sizeof(quant));
		}
		saveToFile.close();
	}
	input.close();
}
#endif
void loadOpeningBookFrom(string opening) {
	float quantize[257];
	for (int i = 0; i < 257; ++i)
	{
		quantize[i] = ((float)i) / 127.0f - 1.0f;
	}
	std::ifstream input(opening, std::ios::binary);
	int cuenta = 0;
	if (input.good())
	{
		std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
		cuenta = (int)buffer.size() / 9;
		if (buffer.size() == 0)
			return;
		opening_book.reserve(130 * (int)buffer.size() / 100);
		for (int i = 0; i < cuenta; ++i)
		{
			uint64_t* Hash = reinterpret_cast<uint64_t*>(&buffer[i * 9]);
			uint8_t quant = buffer[i * 9 + 8];
			float AG = quantize[quant] * 1.2;
			opening_book.emplace(*Hash, AG);
		}
	}
	cerr << "Loaded " << cuenta << " hashes from book" << endl;
}

#endif

int codingame(int argc, char* argv[]) {
	cerr << "CODINGAME v1" << endl;
	VIEW_TREE_DEPTH = 1;
#ifdef OPENING_BOOK
	loadOpeningBookFrom("CGbook.dat");
#endif
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
		gamestate->readTurn(stopwatch);
		cerr << gamestate->WW[0] << " " << gamestate->WW[1] << " ";
		if (player == nullptr)
		{
			player = new MCTS(conf1, &stopwatch, 350 * 1024 * 1024);
		}
		cerr << "Start Search T:" << stopwatch.EllapsedMilliseconds() << "ms" << endl;
		player->Search(model, gamestate, bestMove,&cerr);
		cerr << "End Search Sims:" << SIMCOUNT << " T:" << stopwatch.EllapsedMilliseconds() << "ms";
#ifdef OPENING_BOOK
		cerr << " BOOK HITS:" << BOOKHIT << "/" << BOOKTOTALS << "=" << 100 * BOOKHIT / BOOKTOTALS;
#endif
		cerr << endl;
		cout << PrintMove(bestMove[0], gamestate) << endl;
		//I need to apply my move to calculate my score.
		//Based on my own score I can infer the enemy score at readTurn
		gamestate->Simulate(bestMove);
	}
	return 0;
}

int dumpReplay(int argc, char* argv[]) {
	string replayFile(argv[2]);
	string saveOn(argv[3]);
	vector<_Game> history;

	auto gamestate = std::make_shared<_Game>();
	gamestate->readConfig(stopwatch);
	history.push_back(*gamestate);

	std::ifstream infile(replayFile);
	if (infile.is_open())
	{
		int pos1, pos2;
		float score1, score2;
		infile >> pos1 >> pos2 >> score1 >> score2; infile.ignore();
		vector<int> movelist;
		vector<Move> mm;
		gamestate->getPossibleMoves(0, mm, 0);
		while (!infile.eof())
		{
			int val = -1;
			infile >> val; infile.ignore();
			if (val < 0)
				break;
			movelist.push_back(val);
			Move selected = val;
			bool encontrado = false;
			for (auto& m : mm)
				if ((int)m == val)
				{
					encontrado = true;
					break;
				}
			if (!encontrado)
			{
				cerr << "ERROR MOVES" << endl;
				int T = (int)history.size() - 1;
				cerr << history[T].Print();
				cerr << " POSSIBLE MOVES:[";
				for (auto& m : mm)
					cerr << (int)m << ",";
				cerr << "] Selected" << val << " -> " << (int)selected << endl;


				cerr << endl;
				gamestate->getPossibleMoves(0, mm, 0);
				cerr << "-----------" << endl;
			}
			//

			if (gamestate->isEndGame())
			{
				cerr << "ERROR: Endgame detected before move !!!! Turn:" << gamestate->turn << " Move:" << (int)selected << endl;
			}
			gamestate->Simulate(selected);
			gamestate->getPossibleMoves(0, mm, 0);
			history.push_back(*gamestate);
		}
		assert(history.back().Equals(gamestate));
		gamestate->getPossibleMoves(0, mm, 0);
		if (!gamestate->isEndGame())
		{
			cerr << "ERROR: Endgame not detected!!!!" << endl;
		}
		bool isError = (score1 != gamestate->score0 || score2 != gamestate->score1);
		cerr << "Endgame. Turn:" << (int)gamestate->turn << " Scores:" << (int)gamestate->score0 << "," << (int)gamestate->score1 << " expected " << score1 << "," << score2 << " " << isError << endl;
		float player0Score = (score1 > score2 ? 1.0f : (score1 == score2 ? 0.1f : -1.0f));
		float invScore = (score1 > score2 ? -1.0f : (score1 == score2 ? 0.1f : 1.0f));
		std::ofstream outfile;
		outfile.open(saveOn, std::ios_base::app); // append instead of overwrite
		//outfile.open(saveOn); // append instead of overwrite
		float policy[6];

		if (isError)
		{
			cerr << "es un ERRORORROROR " << replayFile << endl;
			for (int T = 0; T < history.size(); ++T)
			{
				cerr << history[T].Print();
				if (T < movelist.size())
					cerr << " MOVE:" << movelist[T];
				cerr << endl;
				cerr << "-----------" << endl;
			}
		}

		for (int T = 0; T < movelist.size(); ++T)
		{

			{

				float score = history[T].idToPlay == 0 ? player0Score : invScore;
				for (int i = 0; i < 6; ++i) {
					policy[i] = -9.0f;
				}
				{
					_Game SWITCH;
					SWITCH.CopyFrom(history[T]);
					if (SWITCH.idToPlay == 1)
						SWITCH.swapPlayers();
					SWITCH.getPossibleMoves(0, mm, 0);

					//for (int p = 0; p < 2; ++p)
					{
#ifdef _MSC_VER
						outfile << "CELLS0" << ":";
#endif
						for (int i = 0; i < 6; ++i)
						{
							outfile << (int)SWITCH.cell0[i] << " ";
						}
#ifdef _MSC_VER
						outfile << "CELLS0" << ":";
#endif
						for (int i = 0; i < 6; ++i)
						{
							outfile << (int)SWITCH.cell1[i] << " ";
						}
					}
#ifdef _MSC_VER
					outfile << " SCORES:";
#endif


					outfile << (int)SWITCH.score0 << " " << (int)SWITCH.score1 << " ";

#ifdef _MSC_VER
					outfile << " VALUE:";
#endif

					outfile << (score >= 0.0f ? " " : "") << score << " ";
#ifdef _MSC_VER
					outfile << " POLICY:";
#endif


					for (auto& m : mm)
					{
						policy[m] = 0.0f;
					}
					if (policy[movelist[T]] == -9.0f)
					{
						cerr << "SIMULATION ERROR, move not found!!!" << endl;
						cerr << history[T].Print() << endl;
						cerr << SWITCH.Print() << endl;
						cerr << "I have moves [";
						for (auto& m : mm)
						{
							cerr << (int)m << ",";
						}
						cerr << "] Policy:";
						for (int i = 0; i < 6; ++i) {
							cerr << policy[i] << ",";
						}
						cerr << " Replay used:" << movelist[T] << " " << policy[movelist[T]] << endl;
						//abort();
					}
					policy[movelist[T]] = score;

				}


				for (int i = 0; i < 6; ++i)
				{
					outfile << policy[i] << " ";
				}
				outfile << endl;
			}
		}
		outfile.close();
	}
	return 0;
}


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
void halfPrecision() {
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> dis(-1.0f, 1.0f); // rage 0 - 1
	float totERROR = 0.0;
	int countItems = 0;

	for (int l = 0; l < 50000; ++l)
	{
#pragma warning( push )
#pragma warning( disable : 4244 )
		__m256 test = _mm256_set_ps(dis(e2), dis(e2), dis(e2), dis(e2), dis(e2), dis(e2), dis(e2), dis(e2));
#pragma warning( pop )
		__m128i half = _mm256_cvtps_ph(test, _MM_FROUND_NO_EXC);
		__m256 recover = _mm256_cvtph_ps(half);

		float* A = (float*)&test;
		float* B = (float*)&recover;
		for (int i = 0; i < 8; ++i)
		{
			if (l == 0)
				cerr << *A << "," << *B << " = " << (*A - *B) << endl;
			totERROR += abs((*A - *B));
			A++;
			B++;
		}
		countItems += 8;
	}

	cerr << "MeanError:"
		<< std::fixed
		<< std::setprecision(8)

		<< totERROR / (float)(countItems) << endl;
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

void FastTest() {

	pair<string, vector<double>> PP =
	{ "361696461815023644 360292376859642112" , {0.139401 ,0.11889 ,0.470697 ,0.111101 ,0.159912} }
	;
	string uus = PP.first;
	DEBUG_POL[uus] = PP.second;
	MCTS_Conf conf1 = selfPlay_Mode;

	conf1.num_iters_per_turn = 3000;
	conf1.cpuct_base = 3.0f;

	conf1.useHeuristicNN = false;

	conf1.cpuct_inc = 0.0f;
	conf1.cpuct_limit = conf1.cpuct_base;

	conf1.dirichlet_noise_epsilon = 0.0f;
	conf1.dirichlet_noise_alpha = 1.0f;	// Dirichlet alpha = 10 / n --> Max expected moves
	conf1.dirichlet_decay = 0.0f;

	conf1.simpleRandomRange = 0.00f;
	conf1.PROPAGATE_BASE = 0.0f; //Propagate "WIN/LOSS" with 30% at start
	conf1.PROPAGATE_INC = 0.0f; //Linearly increase +70% until endgame
	conf1.POLICY_BACKP_FIRST = 0; //Similarly , but with percentage of turns, first 30% of turns doesn't have any "temperature",
	conf1.POLICY_BACKP_LAST = 0; //from 30% to (100-10=90%) I linearly sharpen policy to get only the best move, 
	MCTS* player1 = new MCTS(conf1, &stopwatch, 400 * 1024 * 1024);
	Game ws = make_shared<_Game>();


	Model model = _Game::CreateNNModel();
	/*cerr << uus << endl;
	stringstream s(uus); // Used for breaking words
	s >> ws->WW[0] >> ws->WW[1];
	*/
	ws->cell0[0] = 1;
	ws->cell0[1] = 1;
	ws->cell0[2] = 0;
	ws->cell0[3] = 0;
	ws->cell0[4] = 3;
	ws->cell0[5] = 1;

	ws->cell1[0] = 0;
	ws->cell1[1] = 2;
	ws->cell1[2] = 1;
	ws->cell1[3] = 0;
	ws->cell1[4] = 0;
	ws->cell1[5] = 0;
	ws->score0 = 24;
	ws->score1 = 15;
	ws->turn = 177;
	ws->idToPlay = 0;
	cerr << ws->Print() << endl;


	/*	ws->Simulate(2);
		ws->swapPlayers();
		ws->swapped = 0;
		ws->idToPlay = 0;
		cerr << ws->Print() << endl;
		ws->score1 = ws->cell0[1] + ws->cell0[2];
		ws->cell0[1] = 0;
		ws->cell0[2] = 0;
		cerr << ws->Print() << endl;*/

	vector<Move> bestMove;
	player1->Search(model, ws, bestMove,&cerr);
	cerr << endl;
}
void tmpValidate() {
	//ValidateModel("candidate.weights");
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


	//halfPrecision();
	string PROGRAMNAME(argv[0]);
	//pruebaNN();
	//	cerr << "PROGRAM NAME IS '" << PROGRAMNAME << "'" << endl;
	//	cerr << "CONSTANT C IS " << CONSTANT_C << endl;
#ifdef _MSC_VER
#ifdef OPENING_BOOK
   //createOpeningBookFrom("DUMP.txt", "openingbook.dat");
	loadOpeningBookFrom("openingbook.dat");
#endif
#endif

	if (argc > 1)
	{
		string tmpOption(argv[1]);

		if (tmpOption == "fasttest")
		{
			FastTest();
			//tmpValidate;
			return 0;
		}

		if (tmpOption == "dumpreplay")
		{
			return dumpReplay(argc, argv);
		}

		//	./bin selfplay <model_filename> <match_count> <threads>
		if (tmpOption == "selfplay")
		{
			return selfPlay(argc, argv);
		}
		if (tmpOption == "pitplay")
		{
			return pitPlay(argc, argv);
		}
	}
	//#ifndef _MSC_VER
	return codingame(argc, argv);
	//#endif
		//return simTest(argc, argv);

}