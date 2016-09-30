#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/numpy.hpp>
#include <boost/make_shared.hpp>
#include <numeric>
#include <flex.h>
#include <flexExt.h>

#ifdef NDEBUG
#	define MODULE_NAME PyNvFlex
#else
#	define MODULE_NAME PyNvFlexD
#endif

namespace bpy = boost::python;
namespace np = boost::numpy;

int size(const np::ndarray &array)
{
	int dims = array.get_nd();
	auto shape = array.get_shape();
	using int__ = std::remove_reference<std::remove_const<decltype(*shape)>::type>::type;
	return int(std::accumulate(shape, shape + dims, int__(1), std::multiplies<int__>()));
}

bpy::object flexErrorCallback;

void _FlexErrorCallback(FlexErrorSeverity type, const char *msg, const char *file, int line)
{
	if (flexErrorCallback)
	{
		flexErrorCallback(type, std::string(msg), std::string(file), line);
	}
}

FlexError PyNvFlexInit(int version, const bpy::object &errorFunc, int deviceIndex)
{
	flexErrorCallback = errorFunc;
	return flexInit(version, _FlexErrorCallback, deviceIndex);
}

using FlexSolver_ = std::shared_ptr<FlexSolver>;
auto CreateFlexSolver = [](FlexSolver *solver)
{
	return FlexSolver_(solver, [](FlexSolver* s) { flexDestroySolver(s); });
};

using FlexTriangleMesh_ = std::shared_ptr<FlexTriangleMesh>;
auto CreateFlexTriangleMesh = [](FlexTriangleMesh *mesh)
{
	return FlexTriangleMesh_(mesh, [](FlexTriangleMesh *mesh) { flexDestroyTriangleMesh(mesh); });
};

using FlexSDF_ = std::shared_ptr<FlexSDF>;
auto CreateFlexSDF = [](FlexSDF *sdf)
{
	return FlexSDF_(sdf, [](FlexSDF *sdf) { flexDestroySDF(sdf); });
};

using FlexExtAsset_ = std::shared_ptr<FlexExtAsset>;
auto CreateFlexExtAsset = [](FlexExtAsset *asset)
{
	return FlexExtAsset_(asset, [](FlexExtAsset *asset) { flexExtDestroyAsset(asset); });
};

using FlexExtContainer_ = std::shared_ptr<FlexExtContainer>;
auto CreateFlexExtContainer = [](FlexExtContainer *container)
{
	return FlexExtContainer_(container, [](FlexExtContainer *container) { flexExtDestroyContainer(container); });
};

using FlexExtInstance_ = std::shared_ptr<FlexExtInstance>;
auto CreateFlexExtInstance = [](const FlexExtContainer_ &container, FlexExtInstance *inst)
{
	return FlexExtInstance_(inst, [=](FlexExtInstance *inst) { flexExtDestroyInstance(container.get(), inst); });
};

FlexSolver_ PyNvFlexCreateSolver(int maxParticles, int maxDiffuseParticles, unsigned char maxNeighborsPerParticle)
{
	return CreateFlexSolver(flexCreateSolver(maxParticles, maxDiffuseParticles, maxNeighborsPerParticle));
}

void PyNvFlexDestroySolver(FlexSolver_ &solver)
{
	solver.reset();
}

void PyNvFlexUpdateSolver(const FlexSolver_ &solver, float dt, int substeps, FlexTimers *timers = nullptr)
{
	if(solver) flexUpdateSolver(solver.get(), dt, substeps, timers);
}

BOOST_PYTHON_FUNCTION_OVERLOADS(flexUpdateSolverOverloads, PyNvFlexUpdateSolver, 3, 4)

np::ndarray FlexParamsGetGravity(const FlexParams &params)
{
	return np::from_data(params.mGravity, np::dtype::get_builtin<float>(), bpy::make_tuple(3), bpy::make_tuple(sizeof(float)), bpy::object());
}

void FlexParamsSetGravity(FlexParams &params, const np::ndarray &gravity)
{
	auto gravityF = gravity.astype(np::dtype::get_builtin<float>());
	auto ptr = reinterpret_cast<float*>(gravityF.get_data());
	std::copy(ptr, ptr + 3, params.mGravity);
}

np::ndarray FlexParamsGetWind(const FlexParams &params)
{
	return np::from_data(params.mWind, np::dtype::get_builtin<float>(), bpy::make_tuple(3), bpy::make_tuple(sizeof(float)), bpy::object());
}

void FlexParamsSetWind(FlexParams &params, const np::ndarray &Wind)
{
	auto windF = Wind.astype(np::dtype::get_builtin<float>());
	auto ptr = reinterpret_cast<float*>(windF.get_data());
	std::copy(ptr, ptr + 3, params.mWind);
}

np::ndarray FlexParamsGetDiffuseSortAxis(const FlexParams &params)
{
	return np::from_data(params.mDiffuseSortAxis, np::dtype::get_builtin<float>(), bpy::make_tuple(3), bpy::make_tuple(sizeof(float)), bpy::object());
}

void FlexParamsSetDiffuseSortAxis(FlexParams &params, const np::ndarray &DiffuseSortAxis)
{
	auto DiffuseSortAxisF = DiffuseSortAxis.astype(np::dtype::get_builtin<float>());
	auto ptr = reinterpret_cast<float*>(DiffuseSortAxisF.get_data());
	std::copy(ptr, ptr + 3, params.mDiffuseSortAxis);
}

np::ndarray FlexParamsGetPlanes(const FlexParams &params)
{
	return np::from_data(params.mPlanes, np::dtype::get_builtin<float>(), bpy::make_tuple(8, 4), bpy::make_tuple(4 * sizeof(float), sizeof(float)), bpy::object());
}

void FlexParamsSetPlanes(FlexParams &params, const np::ndarray &planes)
{
	auto planesF = planes.astype(np::dtype::get_builtin<float>());
	auto ptr = reinterpret_cast<float*>(planesF.get_data());
	std::copy(ptr, ptr + 8 * 4, reinterpret_cast<float*>(params.mPlanes));
}

void FlexSetParams(const FlexSolver_ &solver, const FlexParams &params)
{
	flexSetParams(solver.get(), &params);
}

void FlexGetParams(const FlexSolver_ &solver, FlexParams &params)
{
	flexGetParams(solver.get(), &params);
}

void FlexSetActive(const FlexSolver_ &solver, const np::ndarray &indices)
{
	auto indicesI = indices.astype(np::dtype::get_builtin<int>());
	flexSetActive(solver.get(), reinterpret_cast<int*>(indicesI.get_data()), size(indicesI), eFlexMemoryHost);
}

int FlexGetActiveCount(const FlexSolver_ &solver)
{
	return flexGetActiveCount(solver.get());
}

np::ndarray FlexGetActive(const FlexSolver_ &solver)
{
	int activeCount = FlexGetActiveCount(solver);
	np::ndarray ret = np::zeros(bpy::make_tuple(activeCount), np::dtype::get_builtin<int>());
	flexGetActive(solver.get(), reinterpret_cast<int*>(ret.get_data()), eFlexMemoryHost);
	return ret;
}

void FlexSetParticles(const FlexSolver_ &solver, const np::ndarray &particles)
{
	auto particlesF = particles.astype(np::dtype::get_builtin<float>());
	flexSetParticles(solver.get(), reinterpret_cast<float*>(particlesF.get_data()), size(particlesF) / 4, eFlexMemoryHost);
}

np::ndarray FlexGetParticles(const FlexSolver_ &solver, int n)
{
	np::ndarray particles = np::zeros(bpy::make_tuple(n, 4), np::dtype::get_builtin<float>());
	flexGetParticles(solver.get(), reinterpret_cast<float*>(particles.get_data()), n, eFlexMemoryHost);
	return particles;
}

void FlexSetRestParticles(const FlexSolver_ &solver, const np::ndarray &particles)
{
	auto particlesF = particles.astype(np::dtype::get_builtin<float>());
	flexSetRestParticles(solver.get(), reinterpret_cast<float*>(particlesF.get_data()), size(particlesF) / 4, eFlexMemoryHost);
}

np::ndarray FlexGetSmoothParticles(const FlexSolver_ &solver, int n)
{
	np::ndarray particles = np::zeros(bpy::make_tuple(n, 4), np::dtype::get_builtin<float>());
	flexGetSmoothParticles(solver.get(), reinterpret_cast<float*>(particles.get_data()), n, eFlexMemoryHost);
	return particles;
}

void FlexSetVelocities(const FlexSolver_ &solver, const np::ndarray &velocities)
{
	auto velocitiesF = velocities.astype(np::dtype::get_builtin<float>());
	flexSetVelocities(solver.get(), reinterpret_cast<float*>(velocitiesF.get_data()), size(velocitiesF) / 3, eFlexMemoryHost);
}

np::ndarray FlexGetVelocities(const FlexSolver_ &solver, int n)
{
	np::ndarray velocities = np::zeros(bpy::make_tuple(n, 3), np::dtype::get_builtin<float>());
	flexGetVelocities(solver.get(), reinterpret_cast<float*>(velocities.get_data()), n, eFlexMemoryHost);
	return velocities;
}

void FlexSetPhases(const FlexSolver_ &solver, const np::ndarray &Phases)
{
	auto PhasesF = Phases.astype(np::dtype::get_builtin<int>());
	flexSetPhases(solver.get(), reinterpret_cast<int*>(PhasesF.get_data()), size(PhasesF), eFlexMemoryHost);
}

np::ndarray FlexGetPhases(const FlexSolver_ &solver, int n)
{
	np::ndarray Phases = np::zeros(bpy::make_tuple(n), np::dtype::get_builtin<int>());
	flexGetPhases(solver.get(), reinterpret_cast<int*>(Phases.get_data()), n, eFlexMemoryHost);
	return Phases;
}

void FlexSetSprings(const FlexSolver_ &solver, const np::ndarray &indices, const np::ndarray &restLengths, const np::ndarray &stifness)
{
	auto indices_ = indices.astype(np::dtype::get_builtin<int>());
	auto restLengths_ = restLengths.astype(np::dtype::get_builtin<float>());
	auto stifness_ = stifness.astype(np::dtype::get_builtin<float>());
	flexSetSprings(solver.get(),
		reinterpret_cast<int*>(indices_.get_data()),
		reinterpret_cast<float*>(restLengths_.get_data()),
		reinterpret_cast<float*>(stifness_.get_data()),
		size(indices_) / 2,
		eFlexMemoryHost);
}

bpy::tuple FlexGetSprings(const FlexSolver_ &solver, int n)
{
	auto indices = np::zeros(bpy::make_tuple(n, 2), np::dtype::get_builtin<int>());
	auto restLengths = np::zeros(bpy::make_tuple(n), np::dtype::get_builtin<float>());
	auto stiffness = np::zeros(bpy::make_tuple(n), np::dtype::get_builtin<float>());
	//flexGetSprings(solver.get(),
	//	reinterpret_cast<int*>(indices.get_data()),
	//	reinterpret_cast<float*>(restLengths.get_data()),
	//	reinterpret_cast<float*>(stiffness.get_data()),
	//	n,
	//	eFlexMemoryHost);
	return bpy::make_tuple(indices, restLengths, stiffness);
}

void FlexSetRigids(const FlexSolver_ &solver,
				   const np::ndarray &offsets,
				   const np::ndarray &indices,
				   const np::ndarray &restPositions,
				   const np::ndarray &restNormals,
				   const np::ndarray &stifness,
				   const np::ndarray &rotations,
				   const np::ndarray &translations)
{
	auto offsets_ = offsets.astype(np::dtype::get_builtin<int>());
	auto indices_ = indices.astype(np::dtype::get_builtin<int>());
	auto restPositions_ = restPositions.astype(np::dtype::get_builtin<float>());
	auto restNormals_ = restNormals.astype(np::dtype::get_builtin<float>());
	auto stifness_ = stifness.astype(np::dtype::get_builtin<float>());
	auto rotations_ = rotations.astype(np::dtype::get_builtin<float>());
	auto translations_ = translations.astype(np::dtype::get_builtin<float>());
	flexSetRigids(solver.get(),
		reinterpret_cast<int*>(offsets_.get_data()),
		reinterpret_cast<int*>(indices_.get_data()),
		reinterpret_cast<float*>(restPositions_.get_data()),
		reinterpret_cast<float*>(restNormals_.get_data()),
		reinterpret_cast<float*>(stifness_.get_data()),
		reinterpret_cast<float*>(rotations_.get_data()),
		reinterpret_cast<float*>(translations_.get_data()),
		size(offsets_) - 1,
		eFlexMemoryHost);
}

void FlexSetNormals(const FlexSolver_ &solver, const np::ndarray &normals)
{
	auto normals_ = normals.astype(np::dtype::get_builtin<float>());
	flexSetNormals(solver.get(), reinterpret_cast<float*>(normals_.get_data()), size(normals) / 4, eFlexMemoryHost);
}

np::ndarray FlexGetNormals(const FlexSolver_ &solver, int n)
{
	auto normals = np::zeros(bpy::make_tuple(n, 4), np::dtype::get_builtin<float>());
	flexGetNormals(solver.get(), reinterpret_cast<float*>(normals.get_data()), n, eFlexMemoryHost);
	return normals;
}

bpy::tuple FlexGetRigidTransforms(const FlexSolver_ &solver, int numRigids)
{
	auto rotations = np::zeros(bpy::make_tuple(numRigids, 4), np::dtype::get_builtin<float>());
	auto translations = np::zeros(bpy::make_tuple(numRigids, 3), np::dtype::get_builtin<float>());
	flexGetRigidTransforms(solver.get(),
		reinterpret_cast<float*>(rotations.get_data()),
		reinterpret_cast<float*>(translations.get_data()),
		eFlexMemoryHost);
	return bpy::make_tuple(rotations, translations);
}

FlexTriangleMesh_ FlexCreateTriangleMesh()
{
	return CreateFlexTriangleMesh(flexCreateTriangleMesh());
}

void FlexDestroyTriangleMesh(FlexTriangleMesh_ &mesh)
{
	mesh.reset();
}

void FlexUpdateTriangleMesh(const FlexTriangleMesh_ &mesh,
							const np::ndarray &vertices,
							const np::ndarray &indices,
							const np::ndarray &lower,
							const np::ndarray &upper)
{
	auto vertices_ = vertices.astype(np::dtype::get_builtin<float>());
	auto indices_ = indices.astype(np::dtype::get_builtin<int>());
	auto lower_ = lower.astype(np::dtype::get_builtin<float>());
	auto upper_ = upper.astype(np::dtype::get_builtin<float>());
	flexUpdateTriangleMesh(mesh.get(),
		reinterpret_cast<float*>(vertices_.get_data()),
		reinterpret_cast<int*>(indices_.get_data()),
		size(vertices_), size(indices_) / 3,
		reinterpret_cast<float*>(lower_.get_data()),
		reinterpret_cast<float*>(upper_.get_data()),
		eFlexMemoryHost);
}

bpy::tuple FlexGetTriangleMeshBounds(const FlexTriangleMesh_ &mesh)
{
	auto lower = np::zeros(bpy::make_tuple(3), np::dtype::get_builtin<float>());
	auto upper = np::zeros(bpy::make_tuple(3), np::dtype::get_builtin<float>());
	flexGetTriangleMeshBounds(mesh.get(),
		reinterpret_cast<float*>(lower.get_data()),
		reinterpret_cast<float*>(upper.get_data()));
	return bpy::make_tuple(lower, upper);
}

FlexSDF_ FlexCreateSDF()
{
	return CreateFlexSDF(flexCreateSDF());
}

void FlexDestroySDF(FlexSDF_ &sdf)
{
	sdf.reset();
}

void FlexUpdateSDF(const FlexSDF_ &sdf, const np::ndarray &field)
{
	auto field_ = field.astype(np::dtype::get_builtin<float>());
	flexUpdateSDF(sdf.get(), int(field_.shape(0)), int(field_.shape(1)), int(field_.shape(2)),
		reinterpret_cast<float*>(field_.get_data()), eFlexMemoryHost);
}

void FlexSetShapes(const FlexSolver_ &solver,
				   const bpy::object &geometry,
				   const np::ndarray &shapeAabbMins,
				   const np::ndarray &shapeAabbMaxs,
				   const np::ndarray &shapeOffsets,
				   const np::ndarray &shapePositions,
				   const np::ndarray &shapeRotations,
				   const np::ndarray &shapePrevPositions,
				   const np::ndarray &shapePrevRotations,
				   const np::ndarray &shapeFlags)
{
	bpy::stl_input_iterator<FlexCollisionGeometry> first(geometry), last;
	std::vector<FlexCollisionGeometry> geometry_(first, last);
	auto shapeAabbMins_ = shapeAabbMins.astype(np::dtype::get_builtin<float>());
	auto shapeAabbMaxs_ = shapeAabbMaxs.astype(np::dtype::get_builtin<float>());
	auto shapeOffsets_ = shapeOffsets.astype(np::dtype::get_builtin<int>());
	auto shapePositions_ = shapePositions.astype(np::dtype::get_builtin<float>());
	auto shapeRotations_ = shapeRotations.astype(np::dtype::get_builtin<float>());
	auto shapePrevPositions_ = shapePrevPositions.astype(np::dtype::get_builtin<float>());
	auto shapePrevRotations_ = shapePrevRotations.astype(np::dtype::get_builtin<float>());
	auto shapeFlags_ = shapeFlags.astype(np::dtype::get_builtin<int>());
	flexSetShapes(solver.get(),
		geometry_.data(),
		int(geometry_.size()),
		reinterpret_cast<float*>(shapeAabbMins_.get_data()),
		reinterpret_cast<float*>(shapeAabbMaxs_.get_data()),
		reinterpret_cast<int*>(shapeOffsets_.get_data()),
		reinterpret_cast<float*>(shapePositions_.get_data()),
		reinterpret_cast<float*>(shapeRotations_.get_data()),
		reinterpret_cast<float*>(shapePrevPositions_.get_data()),
		reinterpret_cast<float*>(shapePrevRotations_.get_data()),
		reinterpret_cast<int*>(shapeFlags.get_data()),
		size(shapeOffsets_),
		eFlexMemoryHost);
}

void FlexSetDynamicTriangles(const FlexSolver_ &solver, const np::ndarray &indices, const np::ndarray &normals)
{
	auto indices_ = indices.astype(np::dtype::get_builtin<int>());
	auto normals_ = normals.astype(np::dtype::get_builtin<float>());
	flexSetDynamicTriangles(solver.get(),
		reinterpret_cast<int*>(indices_.get_data()),
		reinterpret_cast<float*>(normals_.get_data()),
		size(indices_) / 3,
		eFlexMemoryHost);
}

bpy::tuple FlexGetDynamicTriangles(const FlexSolver_ &solver, int numTris)
{
	auto indices = np::zeros(bpy::make_tuple(numTris, 3), np::dtype::get_builtin<int>());
	auto normals = np::zeros(bpy::make_tuple(numTris, 3), np::dtype::get_builtin<float>());
	flexGetDynamicTriangles(solver.get(),
		reinterpret_cast<int*>(indices.get_data()),
		reinterpret_cast<float*>(normals.get_data()),
		numTris,
		eFlexMemoryHost);
	return bpy::make_tuple(indices, normals);
}

void FlexSetInflatables(const FlexSolver_ &solver,
						const np::ndarray &startTris,
						const np::ndarray &numTris,
						const np::ndarray &restVolumes,
						const np::ndarray &overPressures,
						const np::ndarray &constraintScales)
{
	auto startTris_ = startTris.astype(np::dtype::get_builtin<int>());
	auto numTris_ = numTris.astype(np::dtype::get_builtin<int>());
	auto restVolumes_ = restVolumes.astype(np::dtype::get_builtin<float>());
	auto overPressures_ = overPressures.astype(np::dtype::get_builtin<float>());
	auto constraintScales_ = constraintScales.astype(np::dtype::get_builtin<float>());
	flexSetInflatables(solver.get(),
		reinterpret_cast<int*>(startTris_.get_data()),
		reinterpret_cast<int*>(numTris_.get_data()),
		reinterpret_cast<float*>(restVolumes_.get_data()),
		reinterpret_cast<float*>(overPressures_.get_data()),
		reinterpret_cast<float*>(constraintScales_.get_data()),
		size(startTris_),
		eFlexMemoryHost);
}

np::ndarray FlexGetDensities(const FlexSolver_ &solver, int maxParticles)
{
	auto densities = np::zeros(bpy::make_tuple(maxParticles), np::dtype::get_builtin<float>());
	flexGetDensities(solver.get(), reinterpret_cast<float*>(densities.get_data()), eFlexMemoryHost);
	return densities;
}

bpy::tuple FlexGetAnisotropy(const FlexSolver_ &solver, int maxParticles)
{
	auto q1 = np::zeros(bpy::make_tuple(4, maxParticles), np::dtype::get_builtin<float>());
	auto q2 = np::zeros(bpy::make_tuple(4, maxParticles), np::dtype::get_builtin<float>());
	auto q3 = np::zeros(bpy::make_tuple(4, maxParticles), np::dtype::get_builtin<float>());
	flexGetAnisotropy(solver.get(),
		reinterpret_cast<float*>(q1.get_data()),
		reinterpret_cast<float*>(q2.get_data()),
		reinterpret_cast<float*>(q2.get_data()),
		eFlexMemoryHost);
	return bpy::make_tuple(q1, q2, q3);
}

bpy::tuple FlexGetDiffuseParticles(const FlexSolver_ &solver, int maxParticles)
{
	auto p = np::zeros(bpy::make_tuple(maxParticles, 4), np::dtype::get_builtin<float>());
	auto v = np::zeros(bpy::make_tuple(maxParticles, 4), np::dtype::get_builtin<float>());
	auto indices = np::zeros(bpy::make_tuple(maxParticles), np::dtype::get_builtin<int>());
	flexGetDiffuseParticles(solver.get(),
		reinterpret_cast<float*>(p.get_data()),
		reinterpret_cast<float*>(v.get_data()),
		reinterpret_cast<int*>(indices.get_data()),
		eFlexMemoryHost);
	return bpy::make_tuple(p, v, indices);
}

void FlexSetDiffuseParticles(const FlexSolver_ &solver, const np::ndarray &p, const np::ndarray &v)
{
	auto p_ = p.astype(np::dtype::get_builtin<float>());
	auto v_ = v.astype(np::dtype::get_builtin<float>());
	flexSetDiffuseParticles(solver.get(),
		reinterpret_cast<float*>(p_.get_data()),
		reinterpret_cast<float*>(v_.get_data()),
		size(p_) / 4,
		eFlexMemoryHost);
}

bpy::tuple FlexGetContacts(const FlexSolver_ &solver, int maxParticles)
{
	auto planes = np::zeros(bpy::make_tuple(maxParticles, 4, 4), np::dtype::get_builtin<float>());
	auto velocities = np::zeros(bpy::make_tuple(maxParticles, 4, 4), np::dtype::get_builtin<float>());
	auto indices = np::zeros(bpy::make_tuple(maxParticles), np::dtype::get_builtin<int>());
	auto counts = np::zeros(bpy::make_tuple(maxParticles), np::dtype::get_builtin<unsigned char>());
	flexGetContacts(solver.get(),
		reinterpret_cast<float*>(planes.get_data()),
		reinterpret_cast<float*>(velocities.get_data()),
		reinterpret_cast<int*>(indices.get_data()),
		reinterpret_cast<unsigned char*>(counts.get_data()),
		eFlexMemoryHost);
	return bpy::make_tuple(planes, velocities, indices, counts);
}

bpy::tuple FlexGetBounds(const FlexSolver_ &solver)
{
	auto low = np::zeros(bpy::make_tuple(3), np::dtype::get_builtin<float>());
	auto high = np::zeros(bpy::make_tuple(3), np::dtype::get_builtin<float>());
	flexGetBounds(solver.get(),
		reinterpret_cast<float*>(low.get_data()),
		reinterpret_cast<float*>(high.get_data()),
		eFlexMemoryHost);
	return bpy::make_tuple(low, high);
}

void FlexPlaneSet(FlexCollisionPlane &plane, const np::ndarray &p)
{
	auto p_ = p.astype(np::dtype::get_builtin<float>());
	auto pp_ = reinterpret_cast<float*>(p_.get_data());
	std::copy(pp_, pp_ + 4, plane.mPlane);
}

np::ndarray FlexPlaneGet(const FlexCollisionPlane &plane)
{
	return np::from_data(plane.mPlane, np::dtype::get_builtin<float>(), bpy::make_tuple(4), bpy::make_tuple(sizeof(float)), bpy::object());
}

FlexTriangleMesh_ FlexTriangleMeshGet(const FlexCollisionTriangleMesh &c)
{
	return FlexTriangleMesh_(c.mMesh, [](FlexTriangleMesh*) {});
}

void FlexTriangleMeshSet(FlexCollisionTriangleMesh &c, const FlexTriangleMesh_ &mesh)
{
	c.mMesh = mesh.get();
}

FlexSDF_ FlexSDFGet(const FlexCollisionSDF &c)
{
	return FlexSDF_(c.mField, [](FlexSDF*) {});
}

void FlexSDFSet(FlexCollisionSDF &c, const FlexSDF_ &sdf)
{
	c.mField = sdf.get();
}

boost::shared_ptr<FlexCollisionGeometry> ConstructFromSphere(const FlexCollisionSphere &sphere)
{
	auto ret = boost::make_shared<FlexCollisionGeometry>();
	ret->mSphere.mRadius = sphere.mRadius;
	return ret;
}

boost::shared_ptr<FlexCollisionGeometry> ConstructFromCapsule(const FlexCollisionCapsule &capsule)
{
	auto ret = boost::make_shared<FlexCollisionGeometry>();
	ret->mCapsule.mRadius = capsule.mRadius;
	ret->mCapsule.mHalfHeight = capsule.mHalfHeight;
	return ret;
}

boost::shared_ptr<FlexCollisionGeometry> ConstructFromPlane(const FlexCollisionPlane &plane)
{
	auto ret = boost::make_shared<FlexCollisionGeometry>();
	std::copy(plane.mPlane, plane.mPlane + 4, ret->mPlane.mPlane);
	return ret;
}

boost::shared_ptr<FlexCollisionGeometry> ConstructFromMesh(const FlexCollisionTriangleMesh &mesh)
{
	auto ret = boost::make_shared<FlexCollisionGeometry>();
	ret->mTriMesh.mMesh = mesh.mMesh;
	ret->mTriMesh.mScale = mesh.mScale;
	return ret;
}

boost::shared_ptr<FlexCollisionGeometry> ConstructFromSDF(const FlexCollisionSDF &sdf)
{
	auto ret = boost::make_shared<FlexCollisionGeometry>();
	ret->mSDF.mField = sdf.mField;
	ret->mSDF.mScale = sdf.mScale;
	return ret;
}

bpy::tuple FlexExtCreateWeldedMeshIndices(const np::ndarray &vertices, float threshold)
{
	auto verticesF = vertices.astype(np::dtype::get_builtin<float>());
	auto V = size(verticesF) / 3;

	auto uniqueVerts = np::zeros(bpy::make_tuple(V), np::dtype::get_builtin<int>());
	auto originalToUniqueMap = np::zeros(bpy::make_tuple(V), np::dtype::get_builtin<int>());

	int count = flexExtCreateWeldedMeshIndices(
		reinterpret_cast<float*>(verticesF.get_data()),
		int(V),
		reinterpret_cast<int*>(uniqueVerts.get_data()),
		reinterpret_cast<int*>(originalToUniqueMap.get_data()),
		threshold);

	return bpy::make_tuple(
		uniqueVerts.slice(0, count),
		originalToUniqueMap.slice(0, count)
	);
}

FlexExtAsset_ FlexExtCreateRigidFromMesh(const np::ndarray &vertices,
										 const np::ndarray &indices,
										 float radius, float expand)
{
	auto verticesF = vertices.astype(np::dtype::get_builtin<float>());
	auto indicesI = indices.astype(np::dtype::get_builtin<int>());
	auto vertexData = reinterpret_cast<float*>(verticesF.get_data());
	auto triangleData = reinterpret_cast<int*>(indicesI.get_data());
	return CreateFlexExtAsset(
		flexExtCreateRigidFromMesh(
			vertexData,
			size(verticesF) / 3,
			triangleData,
			size(indicesI),
			radius, expand
		)
	);
}

FlexExtContainer_ FlexExtCreateContainer(const FlexSolver_ &solver, int maxParticles)
{
	return CreateFlexExtContainer(flexExtCreateContainer(solver.get(), maxParticles));
}

np::ndarray FlexExtAllocParticles(const FlexExtContainer_ &container, int n)
{
	auto indices = np::zeros(bpy::make_tuple(n), np::dtype::get_builtin<int>());
	flexExtAllocParticles(container.get(), n, reinterpret_cast<int*>(indices.get_data()));
	return indices;
}

void FlexExtFreeParticles(const FlexExtContainer_ &container, const np::ndarray &indices)
{
	auto indicesI = indices.astype(np::dtype::get_builtin<int>());
	flexExtFreeParticles(container.get(), size(indicesI), reinterpret_cast<int*>(indicesI.get_data()));
}

FlexExtInstance_ FlexExtCreateInstance(const FlexExtContainer_ &container,
									   const FlexExtAsset_ &asset,
									   const np::ndarray &transform,
									   float vx, float vy, float vz,
									   int phase,
									   float invMassScale)
{
	auto transformF = transform.astype(np::dtype::get_builtin<float>());
	return CreateFlexExtInstance(container,
		flexExtCreateInstance(
			container.get(),
			asset.get(),
			reinterpret_cast<float*>(transformF.get_data()),
			vx, vy, vz, phase, invMassScale
		)
	);
}

void FlexExtTickContainer(const FlexExtContainer_ &container,
						  float dt,
						  int numSubsteps,
						  FlexTimers *timers = nullptr)
{
	flexExtTickContainer(container.get(), dt, numSubsteps, timers);
}

int FlexExtAssetNumParticles(const FlexExtAsset_ &asset)
{
	return asset->mNumParticles;
}

int FlexExtInstanceNumParticles(const FlexExtInstance_ &inst)
{
	return inst->mNumParticles;
}

np::ndarray FlexExtInstanceParticleIndices(const FlexExtInstance_ &inst)
{
	auto N = FlexExtInstanceNumParticles(inst);
	return np::from_data(inst->mParticleIndices, np::dtype::get_builtin<int>(), bpy::make_tuple(N), bpy::make_tuple(sizeof(int)), bpy::object());
}

FlexExtAsset_ FlexExtInstanceAsset(const FlexExtInstance_ &inst)
{
	return FlexExtAsset_(const_cast<FlexExtAsset*>(inst->mAsset), [](FlexExtAsset*) {});
}

np::ndarray FlexExtInstanceTranslation(const FlexExtInstance_ &inst)
{
	return np::from_data(inst->mShapeTranslations, np::dtype::get_builtin<float>(), bpy::make_tuple(3), bpy::make_tuple(sizeof(float)), bpy::object());
}

np::ndarray FlexExtInstanceRotation(const FlexExtInstance_ &inst)
{
	return np::from_data(inst->mShapeRotations, np::dtype::get_builtin<float>(), bpy::make_tuple(4), bpy::make_tuple(sizeof(float)), bpy::object());
}

BOOST_PYTHON_FUNCTION_OVERLOADS(FlexExtTickContainerOverloads, FlexExtTickContainer, 3, 4)

BOOST_PYTHON_MODULE(MODULE_NAME)
{
	using bpy::arg;
	using bpy::args;

	np::initialize();

	bpy::enum_<FlexRelaxationMode>("FlexRelaxationMode")
		.value("eFlexRelaxationGlobal", eFlexRelaxationGlobal)
		.value("eFlexRelaxationLocal", eFlexRelaxationLocal)
		;

	bpy::enum_<FlexPhase>("FlexPhase")
		.value("eFlexPhaseGroupMask", eFlexPhaseGroupMask)
		.value("eFlexPhaseSelfCollide", eFlexPhaseSelfCollide)
		.value("eFlexPhaseSelfCollideFilter", eFlexPhaseSelfCollideFilter)
		.value("eFlexPhaseFluid", eFlexPhaseFluid)
		;

	bpy::enum_<FlexError>("FlexError")
		.value("eFlexErrorNone", eFlexErrorNone)
		.value("eFlexErrorWrongVersion", eFlexErrorWrongVersion)
		.value("eFlexErrorInsufficientGPU", eFlexErrorInsufficientGPU)
		.value("eFlexErrorDriverFailure", eFlexErrorDriverFailure)
		;

	bpy::enum_<FlexMemory>("FlexMemory")
		.value("eFlexMemoryHost", eFlexMemoryHost)
		.value("eFlexMemoryDevice", eFlexMemoryDevice)
		.value("eFlexMemoryHostAsync", eFlexMemoryHostAsync)
		.value("eFlexMemoryDeviceAsync", eFlexMemoryDeviceAsync)
		;

	bpy::enum_<FlexSolverCallbackStage>("FlexSolverCallbackStage")
		.value("eFlexStageIterationStart", eFlexStageIterationStart)
		.value("eFlexStageIterationEnd", eFlexStageIterationEnd)
		.value("eFlexStageSubstepBegin", eFlexStageSubstepBegin)
		.value("eFlexStageSubstepEnd", eFlexStageSubstepEnd)
		.value("eFlexStageUpdateEnd", eFlexStageUpdateEnd)
		.value("eFlexStageCount", eFlexStageCount)
		;

	bpy::enum_<FlexErrorSeverity>("FlexErrorSeverity")
		.value("eFlexLogError", eFlexLogError)
		.value("eFlexLogInfo ", eFlexLogError)
		.value("eFlexLogWarning ", eFlexLogError)
		.value("eFlexLogDebug ", eFlexLogError)
		.value("eFlexLogAll ", eFlexLogError)
		;

	bpy::enum_<FlexCollisionShapeType>("FlexCollisionShapeType")
		.value("eFlexShapeSphere", eFlexShapeSphere)
		.value("eFlexShapeCapsule", eFlexShapeCapsule)
		.value("eFlexShapeConvexMesh", eFlexShapeConvexMesh)
		.value("eFlexShapeTriangleMesh", eFlexShapeTriangleMesh)
		.value("eFlexShapeSDF", eFlexShapeSDF)
		;

	bpy::enum_<FlexCollisionShapeFlags>("FlexCollisionShapeFlags")
		.value("eFlexShapeFlagTypeMask", eFlexShapeFlagTypeMask)
		.value("eFlexShapeFlagDynamic", eFlexShapeFlagDynamic)
		.value("eFlexShapeFlagTrigger", eFlexShapeFlagTrigger)
		.value("eFlexShapeFlagReserved", eFlexShapeFlagReserved)
		;

	bpy::class_<FlexCollisionSphere>("FlexCollisionSphere")
		.def_readwrite("mRadius", &FlexCollisionSphere::mRadius)
		;

	bpy::class_<FlexCollisionCapsule>("FlexCollisionCapsule")
		.def_readwrite("mRadius", &FlexCollisionCapsule::mRadius)
		.def_readwrite("mHalfHeight", &FlexCollisionCapsule::mHalfHeight)
		;

	bpy::class_<FlexCollisionPlane>("FlexCollisionPlane")
		.add_property("mPlane", FlexPlaneGet, FlexPlaneSet)
		;

	bpy::class_<FlexCollisionTriangleMesh>("FlexCollisionTriangleMesh")
		.def_readwrite("mScale", &FlexCollisionTriangleMesh::mScale)
		.add_property("mMesh", FlexTriangleMeshGet, FlexTriangleMeshSet)
		;

	bpy::class_<FlexCollisionSDF>("FlexCollisionSDF")
		.def_readwrite("mScale", &FlexCollisionSDF::mScale)
		.add_property("mField", FlexSDFGet, FlexSDFSet)
		;

	bpy::class_<FlexCollisionGeometry>("FlexCollisionGeometry")
		.def("__init__", bpy::make_constructor(ConstructFromSphere))
		.def("__init__", bpy::make_constructor(ConstructFromCapsule))
		.def("__init__", bpy::make_constructor(ConstructFromPlane))
		.def("__init__", bpy::make_constructor(ConstructFromMesh))
		.def("__init__", bpy::make_constructor(ConstructFromSDF))
		;

	bpy::def("flexMakePhase", flexMakePhase, (arg("group"), arg("flags")));
	bpy::def("flexInit", PyNvFlexInit,
		(arg("version") = FLEX_VERSION, arg("errorFunc") = bpy::object(), arg("deviceIndex") = -1));
	bpy::def("flexShutdown", flexShutdown);
	bpy::def("flexGetVersion", flexGetVersion);

	bpy::class_<FlexSolver_>("FlexSolver", bpy::no_init)
		;

	bpy::def("flexCreateSolver", PyNvFlexCreateSolver, (arg("maxParticles"), arg("maxDiffuseParticles"), arg("maxNeigborsPerParticle") = 96));
	bpy::def("flexDestroySolver", PyNvFlexDestroySolver, (arg("solver")));

	bpy::class_<FlexTimers>("FlexTimers")
		.def_readwrite("mPredict", &FlexTimers::mPredict)
		.def_readwrite("mCreateCellIndices", &FlexTimers::mCreateCellIndices)
		.def_readwrite("mSortCellIndices", &FlexTimers::mSortCellIndices)
		.def_readwrite("mCreateGrid", &FlexTimers::mCreateGrid)
		.def_readwrite("mReorder", &FlexTimers::mReorder)
		.def_readwrite("mCollideParticles", &FlexTimers::mCollideParticles)
		.def_readwrite("mCollideShapes", &FlexTimers::mCollideShapes)
		.def_readwrite("mCollideTriangles", &FlexTimers::mCollideTriangles)
		.def_readwrite("mCollideFields", &FlexTimers::mCollideFields)
		.def_readwrite("mCalculateDensity", &FlexTimers::mCalculateDensity)
		.def_readwrite("mSolveDensities", &FlexTimers::mSolveDensities)
		.def_readwrite("mSolveVelocities", &FlexTimers::mSolveVelocities)
		.def_readwrite("mSolveShapes", &FlexTimers::mSolveShapes)
		.def_readwrite("mSolveSprings", &FlexTimers::mSolveSprings)
		.def_readwrite("mSolveContacts", &FlexTimers::mSolveContacts)
		.def_readwrite("mSolveInflatables", &FlexTimers::mSolveInflatables)
		.def_readwrite("mCalculateAnisotropy", &FlexTimers::mCalculateAnisotropy)
		.def_readwrite("mUpdateDiffuse", &FlexTimers::mUpdateDiffuse)
		.def_readwrite("mUpdateTriangles", &FlexTimers::mUpdateTriangles)
		.def_readwrite("mUpdateNormals", &FlexTimers::mUpdateNormals)
		.def_readwrite("mFinalize", &FlexTimers::mFinalize)
		.def_readwrite("mUpdateBounds", &FlexTimers::mUpdateBounds)
		.def_readwrite("mTotal", &FlexTimers::mTotal)
		;

	bpy::def("flexUpdateSolver", PyNvFlexUpdateSolver, flexUpdateSolverOverloads(args("solver", "dt", "substeps", "timers")));

	bpy::class_<FlexParams>("FlexParams")
		.def_readwrite("mNumIterations", &FlexParams::mNumIterations)
		.add_property("mGravity", FlexParamsGetGravity, FlexParamsSetGravity)
		.def_readwrite("mRadius", &FlexParams::mRadius)
		.def_readwrite("mSolidRestDistance", &FlexParams::mSolidRestDistance)
		.def_readwrite("mFluidRestDistance", &FlexParams::mFluidRestDistance)
		.def_readwrite("mDynamicFriction", &FlexParams::mDynamicFriction)
		.def_readwrite("mStaticFriction", &FlexParams::mStaticFriction)
		.def_readwrite("mParticleFriction", &FlexParams::mParticleFriction)
		.def_readwrite("mRestitution", &FlexParams::mRestitution)
		.def_readwrite("mAdhesion", &FlexParams::mAdhesion)
		.def_readwrite("mSleepThreshold", &FlexParams::mSleepThreshold)
		.def_readwrite("mMaxSpeed", &FlexParams::mMaxSpeed)
		.def_readwrite("mShockPropagation", &FlexParams::mShockPropagation)
		.def_readwrite("mDissipation", &FlexParams::mDissipation)
		.def_readwrite("mDamping", &FlexParams::mDamping)
		.def_readwrite("mInertiaBias", &FlexParams::mInertiaBias)
		.add_property("mWind", FlexParamsGetWind, FlexParamsSetWind)
		.def_readwrite("mDrag", &FlexParams::mDrag)
		.def_readwrite("mLift", &FlexParams::mLift)
		.def_readwrite("mFluid", &FlexParams::mFluid)
		.def_readwrite("mCohesion", &FlexParams::mCohesion)
		.def_readwrite("mSurfaceTension", &FlexParams::mSurfaceTension)
		.def_readwrite("mViscosity", &FlexParams::mViscosity)
		.def_readwrite("mVorticityConfinement", &FlexParams::mVorticityConfinement)
		.def_readwrite("mAnisotropyScale", &FlexParams::mAnisotropyScale)
		.def_readwrite("mAnisotropyMin", &FlexParams::mAnisotropyMin)
		.def_readwrite("mAnisotropyMax", &FlexParams::mAnisotropyMax)
		.def_readwrite("mSmoothing", &FlexParams::mSmoothing)
		.def_readwrite("mSolidPressure", &FlexParams::mSolidPressure)
		.def_readwrite("mFreeSurfaceDrag", &FlexParams::mFreeSurfaceDrag)
		.def_readwrite("mBuoyancy", &FlexParams::mBuoyancy)
		.def_readwrite("mDiffuseThreshold", &FlexParams::mDiffuseThreshold)
		.def_readwrite("mDiffuseBuoyancy", &FlexParams::mDiffuseBuoyancy)
		.def_readwrite("mDiffuseDrag", &FlexParams::mDiffuseDrag)
		.def_readwrite("mDiffuseBallistic", &FlexParams::mDiffuseBallistic)
		.add_property("mDiffuseSortAxis", FlexParamsGetDiffuseSortAxis, FlexParamsSetDiffuseSortAxis)
		.def_readwrite("mDiffuseLifetime", &FlexParams::mDiffuseLifetime)
		.def_readwrite("mPlasticThreshold", &FlexParams::mPlasticThreshold)
		.def_readwrite("mPlasticCreep", &FlexParams::mPlasticCreep)
		.def_readwrite("mCollisionDistance", &FlexParams::mCollisionDistance)
		.def_readwrite("mParticleCollisionMargin", &FlexParams::mParticleCollisionMargin)
		.def_readwrite("mShapeCollisionMargin", &FlexParams::mShapeCollisionMargin)
		.add_property("mPlanes", FlexParamsGetPlanes, FlexParamsSetPlanes)
		.def_readwrite("mNumPlanes", &FlexParams::mNumPlanes)
		.def_readwrite("mRelaxationMode", &FlexParams::mRelaxationMode)
		.def_readwrite("mRelaxationFactor", &FlexParams::mRelaxationFactor)
		;

	bpy::def("flexSetParams", FlexSetParams, (arg("solver"), arg("params")));
	bpy::def("flexGetParams", FlexGetParams, (arg("solver"), arg("params")));
	bpy::def("flexSetActive", FlexSetActive, (arg("solver"), arg("indices")));
	bpy::def("flexGetActive", FlexGetActive, (arg("solver")));
	bpy::def("flexGetActiveCount", FlexGetActiveCount, (arg("solver")));
	bpy::def("flexSetParticles", FlexSetParticles, (arg("solver"), arg("particles")));
	bpy::def("flexGetParticles", FlexGetParticles, (arg("solver"), arg("n")));
	bpy::def("flexSetRestParticles", FlexSetRestParticles, (arg("solver"), arg("particles")));
	bpy::def("flexGetSmoothParticles", FlexGetSmoothParticles, (arg("solver"), arg("n")));
	bpy::def("flexSetVelocities", FlexSetVelocities, (arg("solver"), arg("velocities")));
	bpy::def("flexGetVelocities", FlexGetVelocities, (arg("solver"), arg("n")));
	bpy::def("flexSetPhases", FlexSetPhases, (arg("solver"), arg("phases")));
	bpy::def("flexGetPhases", FlexGetPhases, (arg("solver"), arg("n")));
	bpy::def("flexSetSprings", FlexSetSprings, (arg("solver"), arg("indices"), arg("restLengths"), arg("stifness")));
	bpy::def("flexGetSprings", FlexGetSprings, (arg("solver"), arg("n")));
	bpy::def("flexSetRigids", FlexSetRigids, (arg("solver"), arg("offsets"), arg("indices"), arg("restPositions"), arg("restNormals"), arg("stifness"), arg("rotations"), arg("translations")));
	bpy::def("flexSetNormals", FlexSetNormals, (arg("solver"), arg("normals")));
	bpy::def("flexGetNormals", FlexGetNormals, (arg("solver"), arg("n")));
	bpy::def("flexGetRigidTransforms", FlexGetRigidTransforms, (arg("solver"), arg("numRigids")));

	bpy::class_<FlexTriangleMesh_>("FlexTriangleMesh", bpy::no_init)
		;

	bpy::def("flexCreateTriangleMesh", FlexCreateTriangleMesh);
	bpy::def("flexDestroyTriangleMesh", FlexDestroyTriangleMesh, (arg("mesh")));
	bpy::def("flexUpdateTriangleMesh", FlexUpdateTriangleMesh, (arg("mesh"), arg("vertices"), arg("indices"), arg("lower"), arg("upper")));
	bpy::def("flexGetTriangleMeshBounds", FlexGetTriangleMeshBounds, (arg("mesh")));

	bpy::class_<FlexSDF_>("FlexSDF", bpy::no_init)
		;

	bpy::def("flexCreateSDF", FlexCreateSDF);
	bpy::def("flexDestroySDF", FlexDestroySDF, (arg("sdf")));
	bpy::def("flexUpdateSDF", FlexUpdateSDF, (arg("sdf"), arg("field")));

	bpy::def("flexMakeShapeFlags", flexMakeShapeFlags, (arg("type"), arg("dynamic")));
	bpy::def("flexSetShapes", FlexSetShapes, (arg("solver"), arg("geometry"), arg("shapeAabbMins"), arg("shapeAabbMaxs"), arg("shapeOffsets"), arg("shapePositions"), arg("shapeRotations"), arg("shapePrevPositions"), arg("shapePrevRotations"), arg("shapeFlags")));
	bpy::def("flexSetDynamicTriangles", FlexSetDynamicTriangles, (arg("solver"), arg("indices"), arg("normals")));
	bpy::def("flexGetDynamicTriangles", FlexGetDynamicTriangles, (arg("solver"), arg("numTris")));
	bpy::def("flexSetInflatables", FlexSetInflatables, (arg("solver"), arg("startTris"), arg("numTris"), arg("restVolumes"), arg("overPressures"), arg("constraintScales")));
	bpy::def("flexGetDensities", FlexGetDensities, (arg("solver"), arg("maxParticles")));
	bpy::def("flexGetAnisotropy", FlexGetAnisotropy, (arg("solver"), arg("maxParticles")));
	bpy::def("flexGetDiffuseParticles", FlexGetDiffuseParticles, (arg("solver"), arg("maxParticles")));
	bpy::def("flexSetDiffuseParticles", FlexSetDiffuseParticles, (arg("solver"), arg("p"), arg("v")));
	bpy::def("flexGetContacts", FlexGetContacts, (arg("solver"), arg("maxParticles")));
	bpy::def("flexGetBounds", FlexGetBounds, (arg("solver")));

	bpy::def("flexExtCreateWeldedMeshIndices", FlexExtCreateWeldedMeshIndices, (arg("vertices"), arg("threshold")));

	bpy::class_<FlexExtAsset_>("FlexExtAsset")
		.add_property("mNumParticles", FlexExtAssetNumParticles)
		;

	bpy::class_<FlexExtContainer_>("FlexExtContainer");

	bpy::class_<FlexExtInstance_>("FlexExtInstance")
		.add_property("mNumParticles", FlexExtInstanceNumParticles)
		.add_property("mParticleIndices", FlexExtInstanceParticleIndices)
		.add_property("mAsset", FlexExtInstanceAsset)
		.add_property("mShapeTranslations", FlexExtInstanceTranslation)
		.add_property("mShapeRotations", FlexExtInstanceRotation)
		;

	bpy::def("flexExtCreateRigidFromMesh", FlexExtCreateRigidFromMesh, (arg("vertices"), arg("indices"), arg("radius"), arg("expand")));
	bpy::def("flexExtCreateContainer", FlexExtCreateContainer, (arg("solver"), arg("maxParticles")));
	bpy::def("flexExtAllocParticles", FlexExtAllocParticles, (arg("container"), arg("n")));
	bpy::def("flexExtFreeParticles", FlexExtFreeParticles, (arg("container"), arg("indices")));
	bpy::def("flexExtCreateInstance", FlexExtCreateInstance, (arg("container"), arg("asset"), arg("transform"), arg("vx"), arg("vy"), arg("vz"), arg("phase"), arg("invMassScale")));
	bpy::def("flexExtTickContainer", FlexExtTickContainer, FlexExtTickContainerOverloads(args("container", "dt", "numSubsteps", "timers")));
}