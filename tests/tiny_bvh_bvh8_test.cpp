// Regression tests for the BVH8_CPU traversal kernels.
//
// The first test builds one interior node by hand and traces through every
// combination of hit children in all eight ray octants, with sixteen distance
// orders, five ray lengths, and with and without opacity maps, so that each
// path of the child selection is taken. The second test compares traversal of
// built trees against brute-force double precision triangle tests.

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

using namespace tinybvh;

using Vec3 = BVH8_CPU::Vec3;
using Vertex = BVH8_CPU::Vertex;
static constexpr uint32_t INVALID = ~0u;

static int g_failures = 0;

#define CHECK( cond ) do { if (!(cond)) { if (g_failures++ < 10) printf( "FAIL: %s (line %i)\n", #cond, __LINE__ ); } } while (0)

static void TestMasks()
{
	BVH8_CPU bvh;
	constexpr size_t nodeBlocks = sizeof( BVH8_CPU::BVHNode ) / 64, leafBlocks = sizeof( BVH8_CPU::BVHTri4Leaf ) / 64;
	bvh.bvh8Data = (BVH8_CPU::CacheLine*)malloc64( (nodeBlocks + 8 * leafBlocks) * 64 );
	BVH8_CPU::BVHNode* node = (BVH8_CPU::BVHNode*)bvh.bvh8Data;
	std::mt19937 rng( 17 );
	uint32_t opacity[32];
	size_t checks = 0;
	for (unsigned octant = 0; octant < 8; octant++)
	{
		const Vec3 direction( octant & 1 ? .25f : -.25f, octant & 2 ? .5f : -.5f, octant & 4 ? 2.f : -2.f );
		unsigned rank[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
		for (unsigned order = 0; order < 16; order++)
		{
			std::shuffle( rank, rank + 8, rng );
			// the permutation table lists the children farthest first, for every octant
			for (unsigned i = 0; i < 8; i++)
			{
				const unsigned lane = (unsigned)(std::find( rank, rank + 8, 7 - i ) - rank);
				node->perm[i] = 0;
				for (unsigned q = 0; q < 8; q++) node->perm[i] |= lane << (3 * q);
			}
			for (unsigned mask = 0; mask < 256; mask++)
			{
				for (unsigned lane = 0; lane < 8; lane++)
				{
					// child 'lane' is a flat box at distance rank + 1 along the ray, or empty
					const float z = direction.z * (rank[lane] + 1);
					node->xmin[lane] = node->ymin[lane] = -32;
					node->xmax[lane] = node->ymax[lane] = 96;
					node->zmin[lane] = mask & (1u << lane) ? z : 1e30f;
					node->zmax[lane] = mask & (1u << lane) ? z : -1e30f;
					node->child[lane] = BVH8_CPU::LEAF_BIT | (uint32_t)(nodeBlocks + lane * leafBlocks);
					BVH8_CPU::BVHTri4Leaf* leaf = (BVH8_CPU::BVHTri4Leaf*)(bvh.bvh8Data + nodeBlocks + lane * leafBlocks);
					for (unsigned k = 0; k < 4; k++)
					{
						// only one triangle of the leaf is nondegenerate; rotate it through the four slots
						const bool valid = k == order % 4;
						leaf->SetData( Vec3( -32, -32, z ), Vec3( valid ? 128.f : 0.f, 0, 0 ), Vec3( 0, valid ? 128.f : 0.f, 0 ), 4 * lane + k, k );
						opacity[4 * lane + k] = (lane + order) % 3 == 0 ? 0 : 1;
					}
				}
				for (int mapped = 0; mapped < 2; mapped++)
				{
					bvh.SetOpacityMicroMaps( mapped ? opacity : nullptr, mapped ? 1 : 0 );
					for (float maxt : { 0.f, 1.f, 3.5f, 8.f, std::numeric_limits<float>::infinity() })
					{
						unsigned expected = INVALID;
						float t = maxt;
						for (unsigned lane = 0; lane < 8; lane++)
						{
							const unsigned prim = 4 * lane + order % 4;
							if ((mask & (1u << lane)) && (!mapped || opacity[prim]) && rank[lane] + 1 < t)
								t = float( rank[lane] + 1 ), expected = prim;
						}
						Ray input( Vec3( 0 ), direction, maxt );
						input.instIdx = 5u << impl::bvh_inst_shift<uint32_t>;
						input.hit.prim = INVALID;
						Ray ray = input;
						bvh.Intersect( ray );
						CHECK( bvh.IsOccluded( input ) == (expected != INVALID) );
						CHECK( ray.hit.t == t );
						if (expected == INVALID) CHECK( ray.hit.prim == INVALID );
						else
						{
						#if INST_IDX_BITS == 32
							CHECK( ray.hit.prim == expected && ray.hit.inst == input.instIdx );
						#else
							CHECK( ray.hit.prim == expected + input.instIdx );
						#endif
							CHECK( ray.hit.u == (direction.x * t + 32) / 128 );
							CHECK( ray.hit.v == (direction.y * t + 32) / 128 );
						}
						checks++;
					}
				}
			}
		}
	}
	// a triangle exactly at the origin lies outside the open interval, with signed-zero directions
	bvh.SetOpacityMicroMaps( nullptr, 0 );
	node->zmin[0] = node->zmax[0] = 0;
	for (unsigned i = 1; i < 8; i++) node->zmin[i] = 1e30f, node->zmax[i] = -1e30f;
	BVH8_CPU::BVHTri4Leaf* leaf = (BVH8_CPU::BVHTri4Leaf*)(bvh.bvh8Data + nodeBlocks);
	for (unsigned i = 0; i < 4; i++) leaf->SetData( Vec3( -32, -32, 0 ), Vec3( 128, 0, 0 ), Vec3( 0, 128, 0 ), 0, i );
	for (float dz : { -2.f, 2.f })
	{
		Ray ray( Vec3( 0 ), Vec3( -0.f, 0, dz ), 0 );
		ray.hit.prim = INVALID;
		CHECK( !bvh.IsOccluded( ray ) );
		bvh.Intersect( ray );
		CHECK( ray.hit.prim == INVALID && ray.hit.t == 0 );
	}
	printf( "BVH8_CPU: %zu mask/order/interval/opacity checks\n", checks );
}

static void TestBuilt()
{
	std::mt19937 rng( 123 );
	auto rnd = [&]() { return float( rng() >> 8 ) / float( 1u << 24 ); };
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	for (unsigned i = 0; i < 256; i++)
	{
		const Vec3 a( 4 * rnd() - 2, 4 * rnd() - 2, 4 * rnd() - 2 );
		const Vec3 b = a + Vec3( rnd() - .5f, rnd() - .5f, rnd() - .5f );
		const Vec3 c = a + Vec3( rnd() - .5f, rnd() - .5f, rnd() - .5f );
		for (const Vec3& v : { a, b, c }) indices.push_back( (uint32_t)vertices.size() ), vertices.emplace_back( v );
	}
	BVH8_CPU bvh;
	for (int hq = 0; hq < 2; hq++)
	{
		if (hq) bvh.BuildHQ( vertices.data(), indices.data(), 256 ); else bvh.Build( vertices.data(), 256 );
		size_t hits = 0;
		for (unsigned i = 0; i < 30000; i++)
		{
			Ray input( Vec3( 6 * rnd() - 3, 6 * rnd() - 3, 6 * rnd() - 3 ), Vec3( 4 * rnd() - 2, 4 * rnd() - 2, 4 * rnd() - 2 ), i % 2 ? 10.f : .5f );
			// brute force, in double precision, independent of the layout and the traversal
			using D3 = bvh_traits<double>::vec3;
			auto dbl = []( const auto& v ) { return D3( v.x, v.y, v.z ); };
			const D3 o = dbl( input.O ), d = dbl( input.D );
			double t = input.hit.t, u = 0, v = 0;
			uint32_t prim = INVALID;
			for (uint32_t j = 0; j < 256; j++)
			{
				const D3 a = dbl( vertices[3 * j] ), e1 = dbl( vertices[3 * j + 1] ) - a, e2 = dbl( vertices[3 * j + 2] ) - a;
				const D3 h = tinybvh_cross( d, e2 ), s = o - a, q = tinybvh_cross( s, e1 );
				const double det = tinybvh_dot( e1, h );
				if (det == 0) continue;
				const double tu = tinybvh_dot( s, h ) / det, tv = tinybvh_dot( d, q ) / det, tt = tinybvh_dot( e2, q ) / det;
				if (tu >= 0 && tv >= 0 && tu + tv <= 1 && tt > 0 && tt < t) t = tt, u = tu, v = tv, prim = j;
			}
			Ray ray = input;
			ray.hit.prim = INVALID;
			bvh.Intersect( ray );
			CHECK( ray.hit.prim == prim );
			CHECK( bvh.IsOccluded( input ) == (prim != INVALID) );
			if (prim != INVALID)
			{
				CHECK( std::abs( ray.hit.t - t ) < 2e-5 * std::max( 1., t ) );
				CHECK( std::abs( ray.hit.u - u ) < 2e-5 && std::abs( ray.hit.v - v ) < 2e-5 );
				hits++;
			}
		}
		CHECK( hits > 500 );
		printf( "BVH8_CPU %s: 30000 brute-force comparisons (%zu hits)\n", hq ? "SBVH, indexed" : "SAH, soup", hits );
	}
}

int main()
{
	TestMasks();
	TestBuilt();
	if (g_failures) { printf( "%i BVH8 test failures.\n", g_failures ); return 1; }
	printf( "BVH8 tests passed.\n" );
	return 0;
}
