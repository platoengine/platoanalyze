/*
 * SurfacePressureIntegral.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include <Omega_h_vector.hpp>

#include "OmegaHUtilities.hpp"
#include "ImplicitFunctors.hpp"
#include "NaturalBCUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Class for the evaluation of natural boundary condition surface integrals
 * of type: UNIFORM PRESSURE.
 *
 * \tparam SpatialDim   spatial dimension
 * \tparam NumDofs      number degrees of freedom per natural boundary condition force vector
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs=SpatialDim, Plato::OrdinalType DofsPerNode=NumDofs, Plato::OrdinalType DofOffset=0>
class SurfacePressureIntegral
{
private:
    const std::string mSideSetName;                              /*!< side set name */
    const Omega_h::Vector<NumDofs> mFlux;                        /*!< force vector values */
    Plato::LinearTetCubRuleDegreeOne<SpatialDim> mCubatureRule;  /*!< integration rule */

public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    SurfacePressureIntegral(const std::string & aSideSetName, const Omega_h::Vector<NumDofs>& aFlux);

    /***************************************************************************//**
     * \brief Evaluate natural boundary condition surface integrals.
     *
     * \tparam StateScalarType   state forward automatically differentiated (FAD) type
     * \tparam ControlScalarType control FAD type
     * \tparam ConfigScalarType  configuration FAD type
     * \tparam ResultScalarType  result FAD type
     *
     * \param [in]  aMesh     Omega_h mesh database.
     * \param [in]  aMeshSets Omega_h side set database.
     * \param [in]  aState    2-D view of state variables.
     * \param [in]  aControl  2-D view of control variables.
     * \param [in]  aConfig   3-D view of configuration variables.
     * \param [out] aResult   Assembled vector to which the boundary terms will be added
     * \param [in]  aScale    scalar multiplier
     *
    *******************************************************************************/
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void operator()(Omega_h::Mesh* aMesh,
                    const Omega_h::MeshSets &aMeshSets,
                    const Plato::ScalarMultiVectorT<  StateScalarType>& aState,
                    const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                    const Plato::ScalarArray3DT    < ConfigScalarType>& aConfig,
                    const Plato::ScalarMultiVectorT< ResultScalarType>& aResult,
                    Plato::Scalar aScale) const;
};
// class SurfacePressureIntegral

/***************************************************************************//**
 * \brief SurfacePressureIntegral::SurfacePressureIntegral constructor definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
SurfacePressureIntegral<SpatialDim,NumDofs,DofsPerNode,DofOffset>::SurfacePressureIntegral
(const std::string & aSideSetName, const Omega_h::Vector<NumDofs>& aFlux) :
    mSideSetName(aSideSetName),
    mFlux(aFlux),
    mCubatureRule()
{
}
// class SurfacePressureIntegral::SurfacePressureIntegral

/***************************************************************************//**
 * \brief SurfacePressureIntegral::operator() function definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void SurfacePressureIntegral<SpatialDim,NumDofs,DofsPerNode,DofOffset>::operator()
(Omega_h::Mesh* aMesh,
 const Omega_h::MeshSets &aMeshSets,
 const Plato::ScalarMultiVectorT<  StateScalarType>& aState,
 const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
 const Plato::ScalarArray3DT    < ConfigScalarType>& aConfig,
 const Plato::ScalarMultiVectorT< ResultScalarType>& aResult,
 Plato::Scalar aScale) const
{
    printf("START: SurfacePressureIntegral::operator()");
    // get sideset faces
    auto tFaceLocalOrdinals = Plato::get_face_ordinals(aMeshSets, mSideSetName);
    auto tNumFaces = tFaceLocalOrdinals.size();
    printf("NumFaces=%d\n",tNumFaces);

    // get mesh vertices
    auto tFace2Verts = aMesh->ask_verts_of(SpatialDim-1);
    auto tCell2Verts = aMesh->ask_elem_verts();

    // get face to element graph
    auto tFace2eElems = aMesh->ask_up(SpatialDim - 1, SpatialDim);
    auto tFace2Elems_map   = tFace2eElems.a2ab;
    auto tFace2Elems_elems = tFace2eElems.ab2b;

    // get elem to face map
    auto tElem2Faces = aMesh->ask_down(SpatialDim, SpatialDim - 1).ab2b;

    Plato::NodeCoordinate<SpatialDim> tCoords(aMesh);
    Plato::ComputeSurfaceJacobians<SpatialDim> tComputeSurfaceJacobians;
    Plato::ComputeSurfaceIntegralWeight<SpatialDim> tComputeSurfaceIntegralWeight;
    Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<SpatialDim> tCreateFaceLocalNode2ElemLocalNodeIndexMap;
    Plato::ScalarArray3DT<ConfigScalarType> tJacobian("jacobian", tNumFaces, SpatialDim-1, SpatialDim);

    auto tFlux = mFlux;
    auto tNumDofs = NumDofs;
    auto tNodesPerFace = SpatialDim;
    auto tCubatureWeight = mCubatureRule.getCubWeight();
    if(std::isfinite(tCubatureWeight) == false)
    {
        THROWERR("Surface Pressure Integral: A non-finite cubature weight was detected.")
    }
    auto tMultiplier = aScale / tCubatureWeight;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
    {

      auto tFaceOrdinal = tFaceLocalOrdinals[aFaceI];

      // for each element that the face is connected to: (either 1 or 2)
      for( Plato::OrdinalType tElem = tFace2Elems_map[tFaceOrdinal]; tElem < tFace2Elems_map[tFaceOrdinal+1]; ++tElem )
      {
          // create a map from face local node index to elem local node index
          Plato::OrdinalType tLocalNodeOrd[SpatialDim];
          auto tCellOrdinal = tFace2Elems_elems[tElem];
          tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

          ConfigScalarType tWeight(0.0);
          tComputeSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrd, aConfig, tJacobian);
          tComputeSurfaceIntegralWeight(aFaceI, tMultiplier, tJacobian, tWeight);

          // compute unit normal vector
          auto tCellPoints = Plato::local_element_coords<SpatialDim>(tCellOrdinal, tCoords);
          auto tElemFaceOrdinal = Plato::get_face_ordinal<SpatialDim>(tCellOrdinal, tFaceOrdinal, tElem2Faces);
          auto tUnitNormalVec = Plato::unit_normal_vector(tElemFaceOrdinal, tCellPoints);

          // project into aResult workset
          for( Plato::OrdinalType tNode=0; tNode<tNodesPerFace; tNode++)
          {
              for( Plato::OrdinalType tDof=0; tDof<tNumDofs; tDof++)
              {
                  auto tCellDofOrdinal = tLocalNodeOrd[tNode] * DofsPerNode + tDof + DofOffset;
                  aResult(tCellOrdinal,tCellDofOrdinal) += tUnitNormalVec(tDof) * tFlux(tDof) * tWeight;
              }
          }
      }
    }, "surface pressure integral");
    printf("END: SurfacePressureIntegral::operator()");
}
// class SurfacePressureIntegral::operator()

}
// namespace Plato
