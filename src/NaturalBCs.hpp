#pragma once

#include <sstream>

#include "AnalyzeMacros.hpp"
#include "OmegaHUtilities.hpp"
#include "ImplicitFunctors.hpp"
#include "PlatoStaticsTypes.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

namespace Plato 
{

/***************************************************************************//**
 * \brief Natural boundary condition type ENUM
*******************************************************************************/
struct Neumann
{
    enum bc_t
    {
        UNDEFINED = 0,
        UNIFORM = 1,
        UNIFORM_PRESSURE = 2,
        UNIFORM_COMPONENT = 3,
    };
};

/***************************************************************************//**
 * \brief Return natural boundary condition type
 * \param [in] aType natural boundary condition type string
 * \return natural boundary condition type enum
*******************************************************************************/
inline Plato::Neumann::bc_t natural_boundary_condition_type(const std::string& aType)
{
    if(aType == "Uniform")
    {
        return Plato::Neumann::UNIFORM;
    }
    else if(aType == "Uniform Pressure")
    {
        return Plato::Neumann::UNIFORM_PRESSURE;
    }
    else if(aType == "Uniform Component")
    {
        return Plato::Neumann::UNIFORM_COMPONENT;
    }
    else
    {
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition: 'Type' Parameter Keyword: '" << aType.c_str() << "' is not supported.";
        THROWERR(tMsg.str().c_str())
    }
}

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
    auto tJ11 = aJacobian(aFaceOrdinal, 0, 0) * aJacobian(aFaceOrdinal, 0, 0); 
    auto tJ12 = aJacobian(aFaceOrdinal, 0, 1) * aJacobian(aFaceOrdinal, 0, 1);
    aOutput = aMultiplier * sqrt( tJ11 + tJ12 );
}

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
    auto tJ23 = aJacobian(aFaceOrdinal,0,1) * aJacobian(aFaceOrdinal,1,2) - aJacobian(aFaceOrdinal,0,2) * aJacobian(aFaceOrdinal,1,1);
    auto tJ31 = aJacobian(aFaceOrdinal,0,2) * aJacobian(aFaceOrdinal,1,0) - aJacobian(aFaceOrdinal,0,0) * aJacobian(aFaceOrdinal,1,2);
    auto tJ12 = aJacobian(aFaceOrdinal,0,0) * aJacobian(aFaceOrdinal,1,1) - aJacobian(aFaceOrdinal,0,1) * aJacobian(aFaceOrdinal,1,0);
    aOutput = aMultiplier * sqrt(tJ23*tJ23 + tJ31*tJ31 + tJ12*tJ12);
}

/***************************************************************************//**
 * \brief Class for natural boundary condition surface integrals.
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
    const std::string mSideSetName;
    const Omega_h::Vector<NumDofs> mFlux;
    Plato::LinearTetCubRuleDegreeOne<SpatialDim> mCubatureRule;

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

template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
SurfaceLoadIntegral<SpatialDim,NumDofs,DofsPerNode,DofOffset>::SurfaceLoadIntegral
(const std::string & aSideSetName, const Omega_h::Vector<NumDofs>& aFlux) :
    mSideSetName(aSideSetName),
    mFlux(aFlux)
{}

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
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceIndex)
    {

      auto tFaceOrdinal = tFaceLids[aFaceIndex];

      // for each element that the face is connected to: (either 1 or 2)
      for( Plato::OrdinalType tLocalElemOrd = tFace2Elems_map[tFaceOrdinal]; tLocalElemOrd < tFace2Elems_map[tFaceOrdinal+1]; ++tLocalElemOrd )
      {
          // create a map from face local node index to elem local node index
          Plato::OrdinalType tLocalNodeOrd[SpatialDim];
          auto tCellOrdinal = tFace2Elems_elems[tLocalElemOrd];
          tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, aFaceIndex, tCell2Verts, tFace2Verts, tLocalNodeOrd);

          ConfigScalarType tWeight(0.0);
          auto tMultiplier = aScale / tCubatureWeight;
          tComputeSurfaceJacobians(tCellOrdinal, aFaceIndex, tLocalNodeOrd, aConfig, tJacobian);
          tComputeSurfaceIntegralWeight(aFaceIndex, tMultiplier, tJacobian, tWeight);

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

/***************************************************************************//**
 * \brief Class for natural boundary conditions.
 *
 * \tparam SpatialDim   spatial dimension
 * \tparam NumDofs      number degrees of freedom per natural boundary condition force vector
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs=SpatialDim, Plato::OrdinalType DofsPerNode=NumDofs, Plato::OrdinalType DofOffset=0>
class NaturalBC
{
    const std::string mName;         /*!< user-defined load sublist name */
    const std::string mType;         /*!< natural boundary condition type */
    const std::string mSideSetName;  /*!< side set name */
    Omega_h::Vector<NumDofs> mFlux;  /*!< force vector values */

    Plato::LinearTetCubRuleDegreeOne<SpatialDim> mCubatureRule; /*!< linear cubature rule */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aLoadName user-defined name for natural boundary condition sublist
     * \param [in] aSubList  natural boundary condition input parameter sublist
    *******************************************************************************/
    NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>(const std::string & aLoadName, Teuchos::ParameterList &aSubList) :
        mName(aLoadName),
        mType(aSubList.get<std::string>("Type")),
        mSideSetName(aSubList.get<std::string>("Sides")),
        mCubatureRule()
    {
        auto tFlux = aSubList.get<Teuchos::Array<Plato::Scalar>>("Vector");
        for(Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
        {
            mFlux(tDof) = tFlux[tDof];
        }
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    ~NaturalBC(){}

    /***************************************************************************//**
     * \brief Get the contribution to the assembled forcing vector.
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
     * The boundary terms are integrated on the parameterized surface, \f$\phi(\xi,\psi)\f$, according to:
     *  \f{eqnarray*}{
     *    \phi(\xi,\psi)=
     *       \left\{
     *        \begin{array}{ccc}
     *          N_I\left(\xi,\psi\right) x_I &
     *          N_I\left(\xi,\psi\right) y_I &
     *          N_I\left(\xi,\psi\right) z_I
     *        \end{array}
     *       \right\} \\
     *     f^{el}_{Ii} = \int_{\partial\Omega_{\xi}} N_I\left(\xi,\psi\right) t_i
     *          \left|\left|
     *            \frac{\partial\phi}{\partial\xi} \times \frac{\partial\phi}{\partial\psi}
     *          \right|\right| d\xi d\psi
     * \f}
    *******************************************************************************/
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void get(Omega_h::Mesh* aMesh,
             const Omega_h::MeshSets& aMeshSets,
             const Plato::ScalarMultiVectorT<  StateScalarType>&,
             const Plato::ScalarMultiVectorT<ControlScalarType>&,
             const Plato::ScalarArray3DT    < ConfigScalarType>&,
             const Plato::ScalarMultiVectorT< ResultScalarType>&,
             Plato::Scalar aScale) const;

    /***************************************************************************//**
     * \brief Return natural boundary condition sublist name
     * \return sublist name
    *******************************************************************************/
    decltype(mName) const& getSubListName() const { return mName; }

    /***************************************************************************//**
     * \brief Return side set name for this natural boundary condition
     * \return side set name
    *******************************************************************************/
    decltype(mSideSetName) const& getSideSetName() const { return mSideSetName; }

    /***************************************************************************//**
     * \brief Return force vector for this natural boundary condition
     * \return force vector values
    *******************************************************************************/
    decltype(mFlux) getValues() const { return mFlux; }

    /***************************************************************************//**
     * \brief Return natural boundary condition type
     * \return natural boundary condition type
    *******************************************************************************/
    decltype(mType) getType() const { return mType; }
};


/***************************************************************************//**
 * \brief Owner class that contains a vector of NaturalBC objects.
 *
 * \tparam SpatialDim   spatial dimension
 * \tparam NumDofs      number degrees of freedom per natural boundary condition force vector
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs=SpatialDim, Plato::OrdinalType DofsPerNode=NumDofs, Plato::OrdinalType DofOffset=0>
class NaturalBCs
{
// private member data
private:
    /*!< list of natural boundary condition */
    std::vector<std::shared_ptr<NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>>> mBCs;

// private functions
private:
    /***************************************************************************//**
     * \brief Append natural boundary condition to natural boundary condition list.
     *
     * \param  [in] aName    user-defined name for natural boundary condition sublist
     * \param  [in] aSubList natural boundary condition parameter sublist
    *******************************************************************************/
    void appendNaturalBC(const std::string & aName, Teuchos::ParameterList &aSubList);

    /***************************************************************************//**
     * \brief Return natural boundary condition type: uniform.
     *
     * \param  [in] aName    user-defined name for natural boundary condition sublist
     * \param  [in] aSubList natural boundary condition parameter sublist
     *
     * \return shared pointer to an uniform natural boundary condition
    *******************************************************************************/
    std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>>
    setUniformNaturalBC(const std::string & aName, Teuchos::ParameterList &aSubList);

    /***************************************************************************//**
     * \brief Return natural boundary condition type: uniform pressure.
     *
     * \param  [in] aName    user-defined name for natural boundary condition sublist
     * \param  [in] aSubList natural boundary condition parameter sublist
     *
     * \return shared pointer to an uniform pressure natural boundary condition
    *******************************************************************************/
    std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>>
    setUniformPressureNaturalBC(const std::string & aName, Teuchos::ParameterList &aSubList);

    /***************************************************************************//**
     * \brief Return natural boundary condition type: uniform component.
     *
     * \param  [in] aName    user-defined name for natural boundary condition sublist
     * \param  [in] aSubList natural boundary condition parameter sublist
     *
     * \return shared pointer to an uniform component natural boundary condition
    *******************************************************************************/
    std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>>
    setUniformComponentNaturalBC(const std::string & aName, Teuchos::ParameterList &aSubList);

// public functions
public :
    /***************************************************************************//**
     * \brief Constructor that parses and creates a vector of NaturalBC objects.
     * \param [in] aParams input parameter list
    *******************************************************************************/
    NaturalBCs(Teuchos::ParameterList &aParams);

    /***************************************************************************//**
     * \brief Get the contribution to the assembled forcing vector from the owned
     * boundary conditions.
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
    void get( Omega_h::Mesh* aMesh,
              const Omega_h::MeshSets& aMeshSets,
              const Plato::ScalarMultiVectorT<  StateScalarType>&,
              const Plato::ScalarMultiVectorT<ControlScalarType>&,
              const Plato::ScalarArray3DT    < ConfigScalarType>&,
              const Plato::ScalarMultiVectorT< ResultScalarType>&,
              Plato::Scalar aScale = 1.0) const;
};

/***************************************************************************//**
 * \brief NaturalBC::get function definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>::get
(Omega_h::Mesh* aMesh,
 const Omega_h::MeshSets& aMeshSets,
 const Plato::ScalarMultiVectorT<  StateScalarType>& aState,
 const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
 const Plato::ScalarArray3DT    < ConfigScalarType>& aConfig,
 const Plato::ScalarMultiVectorT< ResultScalarType>& aResult,
 Plato::Scalar aScale) const
{
    auto tType = Plato::natural_boundary_condition_type(mType);
    switch(tType)
    {
        case Plato::Neumann::UNIFORM:
        case Plato::Neumann::UNIFORM_COMPONENT:
        {
            Plato::SurfaceLoadIntegral<SpatialDim, NumDofs, DofsPerNode, DofOffset> tSurfaceLoad(mSideSetName, mFlux);
            tSurfaceLoad(aMesh, aMeshSets, aState, aControl, aConfig, aResult, aScale);
            break;
        }
        case Plato::Neumann::UNIFORM_PRESSURE:
        {
            break;
        }
        default:
        {
            std::stringstream tMsg;
            tMsg << "Natural Boundary Condition Type '" << mType.c_str() << "' is NOT supported.";
            THROWERR(tMsg.str().c_str())
        }
    }
}

/***************************************************************************//**
 * \brief NaturalBC::appendNaturalBC function definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
void NaturalBCs<SpatialDim, NumDofs, DofsPerNode, DofOffset>::appendNaturalBC
(const std::string & aName, Teuchos::ParameterList &aSubList)
{
    const auto tType = aSubList.get<std::string>("Type");
    const auto tNeumannType = Plato::natural_boundary_condition_type(tType);
    std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>> tBC;
    switch(tNeumannType)
    {
        case Plato::Neumann::UNIFORM:
        {
            tBC = this->setUniformNaturalBC(aName, aSubList);
            break;
        }
        case Plato::Neumann::UNIFORM_PRESSURE:
        {
            tBC = this->setUniformPressureNaturalBC(aName, aSubList);
            break;
        }
        case Plato::Neumann::UNIFORM_COMPONENT:
        {
            tBC = this->setUniformComponentNaturalBC(aName, aSubList);
            break;
        }
        default:
        {
            std::stringstream tMsg;
            tMsg << "Natural Boundary Condition Type '" << tType.c_str() << "' is NOT supported.";
            THROWERR(tMsg.str().c_str())
        }
    }
    mBCs.push_back(tBC);

    /*
    if ("Uniform" == tType)
    {
        tBC = this->setUniformNaturalBC(aName, aSubList);
    }
    else if ("Uniform Pressure" == tType)
    {
        tBC = this->setUniformPressureNaturalBC(aName, aSubList);
    }
    else if ("Uniform Component" == tType)
    {
        tBC = this->setUniformComponentNaturalBC(aName, aSubList);
    }
    else
    {
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition Type '" << tType.c_str() << "' is NOT supported.";
        THROWERR(tMsg.str().c_str())
    }
    mBCs.push_back(tBC);
    */
}

/***************************************************************************//**
 * \brief NaturalBC::setUniformNaturalBC function definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>>
NaturalBCs<SpatialDim, NumDofs, DofsPerNode, DofOffset>::setUniformNaturalBC
(const std::string & aName, Teuchos::ParameterList &aSubList)
{
    bool tBC_Value = aSubList.isType<Plato::Scalar>("Value");
    bool tBC_Values = aSubList.isType<Teuchos::Array<Plato::Scalar>>("Values");

    const auto tType = aSubList.get < std::string > ("Type");
    std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>> tBC;
    if (tBC_Values && tBC_Value)
    {
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition: 'Values' OR 'Value' Parameter Keyword in "
            << "Parameter Sublist: '" << aName.c_str() << "' is NOT defined.";
        THROWERR(tMsg.str().c_str())
    }
    else if (tBC_Values)
    {
        auto tValues = aSubList.get<Teuchos::Array<Plato::Scalar>>("Values");
        aSubList.set("Vector", tValues);
    }
    else if (tBC_Value)
    {
        Teuchos::Array<Plato::Scalar> tFluxVector(NumDofs, 0.0);
        auto tValue = aSubList.get<Plato::Scalar>("Value");
        tFluxVector[0] = tValue;
        aSubList.set("Vector", tFluxVector);
    }
    else
    {
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition: Uniform Boundary Condition in Parameter Sublist: '"
            << aName.c_str() << "' was NOT parsed.";
        THROWERR(tMsg.str().c_str())
    }

    tBC = std::make_shared<Plato::NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>>(aName, aSubList);
    return tBC;
}

/***************************************************************************//**
 * \brief NaturalBC::setUniformPressureNaturalBC function definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>>
NaturalBCs<SpatialDim, NumDofs, DofsPerNode, DofOffset>::setUniformPressureNaturalBC
(const std::string & aName, Teuchos::ParameterList &aSubList)
{
    if(aSubList.isParameter("Value") == false)
    {
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition: 'Value' Parameter Keyword in "
            << "Parameter Sublist: '" << aName.c_str() << "' is NOT defined.";
        THROWERR(tMsg.str().c_str())
    }
    auto tValue = aSubList.get<Plato::Scalar>("Value");
    Teuchos::Array<Plato::Scalar> tFluxVector(NumDofs, tValue);
    aSubList.set("Vector", tFluxVector);

    auto tBC = std::make_shared<Plato::NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>>(aName, aSubList);
    return tBC;
}

/***************************************************************************//**
 * \brief NaturalBC::setUniformComponentNaturalBC function definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>>
NaturalBCs<SpatialDim, NumDofs, DofsPerNode, DofOffset>::setUniformComponentNaturalBC
(const std::string & aName, Teuchos::ParameterList &aSubList)
{
    if(aSubList.isParameter("Value") == false)
    {
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition: 'Value' Parameter Keyword in "
            << "Parameter Sublist: '" << aName.c_str() << "' is NOT defined.";
        THROWERR(tMsg.str().c_str())
    }
    auto tValue = aSubList.get<Plato::Scalar>("Value");

    if(aSubList.isParameter("Component") == false)
    {
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition: 'Component' Parameter Keyword in "
            << "Parameter Sublist: '" << aName.c_str() << "' is NOT defined.";
        THROWERR(tMsg.str().c_str())
    }
    Teuchos::Array<Plato::Scalar> tFluxVector(NumDofs, 0.0);
    auto tFluxComponent = aSubList.get<std::string>("Component");

    std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>> tBC;
    if( (tFluxComponent == "x" || tFluxComponent == "X") )
    {
        tFluxVector[0] = tValue;
    }
    else
    if( (tFluxComponent == "y" || tFluxComponent == "Y") && DofsPerNode > 1 )
    {
        tFluxVector[1] = tValue;
    }
    else
    if( (tFluxComponent == "z" || tFluxComponent == "Z") && DofsPerNode > 2 )
    {
        tFluxVector[2] = tValue;
    }
    else
    {
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition: 'Component' Parameter Keyword: '" << tFluxComponent.c_str()
            << "' in Parameter Sublist: '" << aName.c_str() << "' is NOT supported. "
            << "Options are: 'X' or 'x', 'Y' or 'y', and 'Z' or 'z'.";
        THROWERR(tMsg.str().c_str())
    }

    aSubList.set("Vector", tFluxVector);
    tBC = std::make_shared<Plato::NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>>(aName, aSubList);
    return tBC;
}

/***************************************************************************//**
 * \brief NaturalBCs Constructor definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
NaturalBCs<SpatialDim, NumDofs, DofsPerNode, DofOffset>::NaturalBCs(Teuchos::ParameterList &aParams) :
mBCs()
{
    for (Teuchos::ParameterList::ConstIterator tItr = aParams.begin(); tItr != aParams.end(); ++tItr)
    {
        const Teuchos::ParameterEntry &tEntry = aParams.entry(tItr);
        if (!tEntry.isList())
        {
            THROWERR("Parameter in Boundary Conditions block not valid.  Expect lists only.")
        }

        const std::string &tName = aParams.name(tItr);
        if(aParams.isSublist(tName) == false)
        {
            std::stringstream tMsg;
            tMsg << "Natural Boundary Condition: Sublist: '" << tName.c_str() << "' is NOT defined.";
            THROWERR(tMsg.str().c_str())
        }
        Teuchos::ParameterList &tSubList = aParams.sublist(tName);

        if(tSubList.isParameter("Type") == false)
        {
            std::stringstream tMsg;
            tMsg << "Natural Boundary Condition: 'Type' Parameter Keyword in Parameter Sublist: '"
                << tName.c_str() << "' is NOT defined.";
            THROWERR(tMsg.str().c_str())
        }

        this->appendNaturalBC(tName, tSubList);
    }
}

/***************************************************************************//**
 * \brief NaturalBCs::get function definition
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void NaturalBCs<SpatialDim,NumDofs,DofsPerNode,DofOffset>::get(Omega_h::Mesh* aMesh,
     const Omega_h::MeshSets& aMeshSets,
     const Plato::ScalarMultiVectorT<  StateScalarType>& aState,
     const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
     const Plato::ScalarArray3DT    < ConfigScalarType>& aConfig,
     const Plato::ScalarMultiVectorT< ResultScalarType>& aResult,
     Plato::Scalar aScale) const
{
    for (const auto &tMyBC : mBCs)
    {
        tMyBC->get(aMesh, aMeshSets, aState, aControl, aConfig, aResult, aScale);
    }
}

}
// namespace Plato

