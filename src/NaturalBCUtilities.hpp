/*
 * NaturalBCUtilities.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include <Omega_h_array.hpp>

#include "alg/PlatoLambda.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Create face local node index to element local node index map
 *
 * \tparam SpatialDim  spatial dimensions
 *
*******************************************************************************/
template<Plato::OrdinalType SpatialDim>
class CreateFaceLocalNode2ElemLocalNodeIndexMap
{
public:
    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    CreateFaceLocalNode2ElemLocalNodeIndexMap(){}

    /***************************************************************************//**
     * \brief Return face local node index to element local node index map
     *
     * \param [in]  aCellOrdinal    cell ordinal
     * \param [in]  aFaceOrdinal    face ordinal
     * \param [in]  aCell2Verts     cell to vertices map
     * \param [in]  aFace2Verts     face to vertices map
     * \param [out] aLocalNodeOrd   face local node index to element local node index map
    *******************************************************************************/
    DEVICE_TYPE inline void operator()
    (const Plato::OrdinalType & aCellOrdinal,
     const Plato::OrdinalType & aFaceOrdinal,
     const Omega_h::LOs & aCell2Verts,
     const Omega_h::LOs & aFace2Verts,
     Plato::OrdinalType aLocalNodeOrd[SpatialDim]) const;
};
// class CreateFaceLocalNode2ElemLocalNodeIndexMap

/***************************************************************************//**
 * \brief Return face local node index to element local node index map (1-D specialization)
*******************************************************************************/
template<>
DEVICE_TYPE inline void
CreateFaceLocalNode2ElemLocalNodeIndexMap<1>::operator()
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::OrdinalType & aFaceOrdinal,
 const Omega_h::LOs & aCell2Verts,
 const Omega_h::LOs & aFace2Verts,
 Plato::OrdinalType aLocalNodeOrd[1]) const
{
    Plato::OrdinalType tNodesPerFace = 1;
    Plato::OrdinalType tNodesPerCell = 2;
    for( Plato::OrdinalType tNodeI = 0; tNodeI < tNodesPerFace; tNodeI++)
    {
        for( Plato::OrdinalType tNodeJ = 0; tNodeJ < tNodesPerCell; tNodeJ++)
        {
            if( aFace2Verts[aFaceOrdinal*tNodesPerFace+tNodeI] == aCell2Verts[aCellOrdinal*tNodesPerCell + tNodeJ] )
            {
                aLocalNodeOrd[tNodeI] = tNodeJ;
            }
        }
    }
}
// class CreateFaceLocalNode2ElemLocalNodeIndexMap<1>::operator()

/***************************************************************************//**
 * \brief Return face local node index to element local node index map (2-D specialization)
*******************************************************************************/
template<>
DEVICE_TYPE inline void
CreateFaceLocalNode2ElemLocalNodeIndexMap<2>::operator()
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::OrdinalType & aFaceOrdinal,
 const Omega_h::LOs & aCell2Verts,
 const Omega_h::LOs & aFace2Verts,
 Plato::OrdinalType aLocalNodeOrd[2]) const
{
    Plato::OrdinalType tNodesPerFace = 2;
    Plato::OrdinalType tNodesPerCell = 3;
    for( Plato::OrdinalType tNodeI = 0; tNodeI < tNodesPerFace; tNodeI++)
    {
        for( Plato::OrdinalType tNodeJ = 0; tNodeJ < tNodesPerCell; tNodeJ++)
        {
            if( aFace2Verts[aFaceOrdinal*tNodesPerFace+tNodeI] == aCell2Verts[aCellOrdinal*tNodesPerCell + tNodeJ] )
            {
                aLocalNodeOrd[tNodeI] = tNodeJ;
            }
        }
    }
}
// class CreateFaceLocalNode2ElemLocalNodeIndexMap<2>::operator()

/***************************************************************************//**
 * \brief Return face local node index to element local node index map (3-D specialization)
*******************************************************************************/
template<>
DEVICE_TYPE inline void
CreateFaceLocalNode2ElemLocalNodeIndexMap<3>::operator()
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::OrdinalType & aFaceOrdinal,
 const Omega_h::LOs & aCell2Verts,
 const Omega_h::LOs & aFace2Verts,
 Plato::OrdinalType aLocalNodeOrd[3]) const
{
    Plato::OrdinalType tNodesPerFace = 3;
    Plato::OrdinalType tNodesPerCell = 4;
    for( Plato::OrdinalType tNodeI = 0; tNodeI < tNodesPerFace; tNodeI++)
    {
        for( Plato::OrdinalType tNodeJ = 0; tNodeJ < tNodesPerCell; tNodeJ++)
        {
            if( aFace2Verts[aFaceOrdinal*tNodesPerFace+tNodeI] == aCell2Verts[aCellOrdinal*tNodesPerCell + tNodeJ] )
            {
                aLocalNodeOrd[tNodeI] = tNodeJ;
            }
        }
    }
}
// class CreateFaceLocalNode2ElemLocalNodeIndexMap<3>::operator()


/***************************************************************************//**
 * \brief Compute surface Jacobians needed for surface integrals.
 *
 * \tparam SpatialDim  spatial dimensions
 *
*******************************************************************************/
template<Plato::OrdinalType SpatialDim>
class ComputeSurfaceJacobians
{
public:
    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    ComputeSurfaceJacobians(){}

    /***************************************************************************//**
     * \brief Compute surface Jacobians.
     *
     * \tparam ConfigScalarType configuration forward automatically differentiation (FAD) type
     * \tparam ResultScalarType output FAD type
     *
     * \param [in]  aCellOrdinal   cell ordinal
     * \param [in]  aFaceOrdinal   face ordinal
     * \param [in]  aLocalNodeOrd  face local node index to element local node index map
     * \param [in]  aConfig        cell to vertices map
     * \param [out] aJacobian      surface Jacobians
     *
    *******************************************************************************/
    template<typename ConfigScalarType, typename ResultScalarType>
    DEVICE_TYPE inline void operator()
    (const Plato::OrdinalType & aCellOrdinal,
     const Plato::OrdinalType & aFaceOrdinal,
     const Plato::OrdinalType aLocalNodeOrd[SpatialDim],
     const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
     const Plato::ScalarArray3DT<ResultScalarType> & aJacobian) const;
};
// class ComputeSurfaceJacobians

/***************************************************************************//**
 * \brief Compute surface Jacobians (1-D specialization)
*******************************************************************************/
template<>
template<typename ConfigScalarType, typename ResultScalarType>
DEVICE_TYPE inline void
ComputeSurfaceJacobians<1>::operator()
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::OrdinalType & aFaceOrdinal,
 const Plato::OrdinalType aLocalNodeOrd[1],
 const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
 const Plato::ScalarArray3DT<ResultScalarType> & aJacobian) const
{ return; }
// class ComputeSurfaceJacobians<1>::operator()

/***************************************************************************//**
 * \brief Compute surface Jacobians (2-D specialization)
*******************************************************************************/
template<>
template<typename ConfigScalarType, typename ResultScalarType>
DEVICE_TYPE inline void
ComputeSurfaceJacobians<2>::operator()
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::OrdinalType & aFaceOrdinal,
 const Plato::OrdinalType aLocalNodeOrd[2],
 const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
 const Plato::ScalarArray3DT<ResultScalarType> & aJacobian) const
{
    const Plato::OrdinalType tSpaceDim = 2;
    const Plato::OrdinalType tSpaceDimMinusOne = 1;
    for( Plato::OrdinalType tNode=0; tNode < tSpaceDimMinusOne; tNode++ )
    {
        for( Plato::OrdinalType tDim=0; tDim < tSpaceDim; tDim++ )
        {
            aJacobian(aFaceOrdinal,tNode,tDim) =
                aConfig(aCellOrdinal, aLocalNodeOrd[tNode], tDim) - aConfig(aCellOrdinal, aLocalNodeOrd[tSpaceDimMinusOne], tDim);
        }
    }
}
// class ComputeSurfaceJacobians<2>::operator()

/***************************************************************************//**
 * \brief Compute surface Jacobians (3-D specialization)
*******************************************************************************/
template<>
template<typename ConfigScalarType, typename ResultScalarType>
DEVICE_TYPE inline void
ComputeSurfaceJacobians<3>::operator()
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::OrdinalType & aFaceOrdinal,
 const Plato::OrdinalType aLocalNodeOrd[3],
 const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
 const Plato::ScalarArray3DT<ResultScalarType> & aJacobian) const
{
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tSpaceDimMinusOne = 2;
    for( Plato::OrdinalType tNode=0; tNode < tSpaceDimMinusOne; tNode++ )
    {
        for( Plato::OrdinalType tDim=0; tDim < tSpaceDim; tDim++ )
        {
            aJacobian(aFaceOrdinal,tNode,tDim) =
                aConfig(aCellOrdinal, aLocalNodeOrd[tNode], tDim) - aConfig(aCellOrdinal, aLocalNodeOrd[tSpaceDimMinusOne], tDim);
        }
    }
}
// class ComputeSurfaceJacobians<3>::operator()


/***************************************************************************//**
 * \brief Compute cubature weight for surface integrals.
 *
 * \tparam SpatialDim  spatial dimensions
 *
*******************************************************************************/
template<Plato::OrdinalType SpatialDim>
class ComputeSurfaceIntegralWeight
{
public:
    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    ComputeSurfaceIntegralWeight(){}

    /***************************************************************************//**
     * \brief Compute cubature weight for surface integrals.
     *
     * \tparam ConfigScalarType configuration forward automatically differentiation (FAD) type
     * \tparam ResultScalarType output FAD type
     *
     * \param [in]  aFaceOrdinal  face ordinal
     * \param [in]  aMultiplier   scalar multiplier
     * \param [in]  aJacobian     surface Jacobians
     * \param [out] aJacobian     cubature weight
     *
    *******************************************************************************/
    template<typename ConfigScalarType, typename ResultScalarType>
    DEVICE_TYPE inline void operator()
    (const Plato::OrdinalType & aFaceOrdinal,
     const Plato::Scalar & aMultiplier,
     const Plato::ScalarArray3DT<ConfigScalarType> & aJacobian,
     ResultScalarType & aOutput) const;
};
// class ComputeSurfaceIntegralWeight

/***************************************************************************//**
 * \brief Compute cubature weight for surface integrals (1-D specialization)
*******************************************************************************/
template<>
template<typename ConfigScalarType, typename ResultScalarType>
DEVICE_TYPE inline void
ComputeSurfaceIntegralWeight<1>::operator()
(const Plato::OrdinalType & aFaceOrdinal,
 const Plato::Scalar & aMultiplier,
 const Plato::ScalarArray3DT<ConfigScalarType> & aJacobian,
 ResultScalarType & aOutput) const
{
    aOutput = aMultiplier;
}
// class ComputeSurfaceIntegralWeight<1>::operator()

/***************************************************************************//**
 * \brief Compute cubature weight for surface integrals (2-D specialization)
*******************************************************************************/
template<>
template<typename ConfigScalarType, typename ResultScalarType>
DEVICE_TYPE inline void
ComputeSurfaceIntegralWeight<2>::operator()
(const Plato::OrdinalType & aFaceOrdinal,
 const Plato::Scalar & aMultiplier,
 const Plato::ScalarArray3DT<ConfigScalarType> & aJacobian,
 ResultScalarType & aOutput) const
{
    ConfigScalarType tJ11 = aJacobian(aFaceOrdinal, 0, 0) * aJacobian(aFaceOrdinal, 0, 0);
    ConfigScalarType tJ12 = aJacobian(aFaceOrdinal, 0, 1) * aJacobian(aFaceOrdinal, 0, 1);
    aOutput = aMultiplier * sqrt( tJ11 + tJ12 );
}
// class ComputeSurfaceIntegralWeight<2>::operator()

/***************************************************************************//**
 * \brief Compute cubature weight for surface integrals (3-D specialization)
*******************************************************************************/
template<>
template<typename ConfigScalarType, typename ResultScalarType>
DEVICE_TYPE inline void
ComputeSurfaceIntegralWeight<3>::operator()
(const Plato::OrdinalType & aFaceOrdinal,
 const Plato::Scalar & aMultiplier,
 const Plato::ScalarArray3DT<ConfigScalarType> & aJacobian,
 ResultScalarType & aOutput) const
{
    ConfigScalarType tJ23 = aJacobian(aFaceOrdinal,0,1) * aJacobian(aFaceOrdinal,1,2) - aJacobian(aFaceOrdinal,0,2) * aJacobian(aFaceOrdinal,1,1);
    ConfigScalarType tJ31 = aJacobian(aFaceOrdinal,0,2) * aJacobian(aFaceOrdinal,1,0) - aJacobian(aFaceOrdinal,0,0) * aJacobian(aFaceOrdinal,1,2);
    ConfigScalarType tJ12 = aJacobian(aFaceOrdinal,0,0) * aJacobian(aFaceOrdinal,1,1) - aJacobian(aFaceOrdinal,0,1) * aJacobian(aFaceOrdinal,1,0);
    aOutput = aMultiplier * sqrt(tJ23*tJ23 + tJ31*tJ31 + tJ12*tJ12);
}
// class ComputeSurfaceIntegralWeight<3>::operator()

}
// namespace Plato
