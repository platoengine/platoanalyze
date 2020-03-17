/*
 * SurfaceLoadIntegral.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include <Omega_h_vector.hpp>

#include "OmegaHUtilities.hpp"
#include "NaturalBCUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Class for the evaluation of natural boundary condition surface integrals
 * of type: UNIFORM and UNIFORM COMPONENT.
 *
 * \tparam SpatialDim   spatial dimension
 * \tparam NumDofs      number degrees of freedom per natural boundary condition force vector
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs=SpatialDim, Plato::OrdinalType DofsPerNode=NumDofs, Plato::OrdinalType DofOffset=0>
class SurfaceLoadIntegral
{
private:
    const std::string mSideSetName;                              /*!< side set name */
    const Omega_h::Vector<NumDofs> mFlux;                        /*!< force vector values */
    Plato::LinearTetCubRuleDegreeOne<SpatialDim> mCubatureRule;  /*!< integration rule */

public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    SurfaceLoadIntegral(const std::string & aSideSetName, const Omega_h::Vector<NumDofs>& aFlux);

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
// class SurfaceLoadIntegral

/***************************************************************************//**
 * \brief SurfaceLoadIntegral::SurfaceLoadIntegral constructor definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
SurfaceLoadIntegral<SpatialDim,NumDofs,DofsPerNode,DofOffset>::SurfaceLoadIntegral
(const std::string & aSideSetName, const Omega_h::Vector<NumDofs>& aFlux) :
    mSideSetName(aSideSetName),
    mFlux(aFlux),
    mCubatureRule()
{
}
// class SurfaceLoadIntegral::SurfaceLoadIntegral

/***************************************************************************//**
 * \brief SurfaceLoadIntegral::operator() function definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void SurfaceLoadIntegral<SpatialDim,NumDofs,DofsPerNode,DofOffset>::operator()
(Omega_h::Mesh* aMesh,
 const Omega_h::MeshSets &aMeshSets,
 const Plato::ScalarMultiVectorT<  StateScalarType>& aState,
 const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
 const Plato::ScalarArray3DT    < ConfigScalarType>& aConfig,
 const Plato::ScalarMultiVectorT< ResultScalarType>& aResult,
 Plato::Scalar aScale) const
{
    // get sideset faces
    auto tFaceLids = Plato::get_face_local_ordinals(aMeshSets, mSideSetName);
    auto tNumFaces = tFaceLids.size();

    // get mesh vertices
    auto tFace2Verts = aMesh->ask_verts_of(SpatialDim-1);
    auto tCell2Verts = aMesh->ask_elem_verts();

    auto tFace2eElems = aMesh->ask_up(SpatialDim - 1, SpatialDim);
    auto tFace2Elems_map   = tFace2eElems.a2ab;
    auto tFace2Elems_elems = tFace2eElems.ab2b;

    Plato::ComputeSurfaceJacobians<SpatialDim> tComputeSurfaceJacobians;
    Plato::ComputeSurfaceIntegralWeight<SpatialDim> tComputeSurfaceIntegralWeight;
    Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<SpatialDim> tCreateFaceLocalNode2ElemLocalNodeIndexMap;
    Plato::ScalarArray3DT<ConfigScalarType> tJacobian("jacobian", tNumFaces, SpatialDim-1, SpatialDim);

    auto tFlux = mFlux;
    auto tNodesPerFace = SpatialDim;
    auto tCubatureWeight = mCubatureRule.getCubWeight();
    if(std::isfinite(tCubatureWeight) == false)
    {
        THROWERR("Natural Boundary Condition: A non-finite cubature weight was detected.")
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceI)
    {

      auto tFaceOrdinal = tFaceLids[aFaceI];

      // for each element that the face is connected to: (either 1 or 2)
      for( Plato::OrdinalType tLocalElemOrd = tFace2Elems_map[tFaceOrdinal]; tLocalElemOrd < tFace2Elems_map[tFaceOrdinal+1]; ++tLocalElemOrd )
      {
          // create a map from face local node index to elem local node index
          Plato::OrdinalType tLocalNodeOrd[SpatialDim];
          auto tCellOrdinal = tFace2Elems_elems[tLocalElemOrd];
          tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

          ConfigScalarType tWeight(0.0);
          auto tMultiplier = aScale / tCubatureWeight;
          tComputeSurfaceJacobians(tCellOrdinal, aFaceI, tLocalNodeOrd, aConfig, tJacobian);
          tComputeSurfaceIntegralWeight(aFaceI, tMultiplier, tJacobian, tWeight);

          // project into aResult workset
          for( Plato::OrdinalType tNode=0; tNode<tNodesPerFace; tNode++)
          {
              for( Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
              {
                  auto tCellDofOrdinal = tLocalNodeOrd[tNode] * DofsPerNode + tDof + DofOffset;
                  aResult(tCellOrdinal,tCellDofOrdinal) += tWeight*tFlux[tDof];
              }
          }
      }
    }, "surface load integral");
}
// class SurfaceLoadIntegral::operator()

}
// namespace Plato
