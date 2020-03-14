#ifndef NATURAL_BC_HPP
#define NATURAL_BC_HPP

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

/******************************************************************************/
/*!
  \brief Class for natural boundary conditions.
*/
  template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs=SpatialDim, Plato::OrdinalType DofsPerNode=NumDofs, Plato::OrdinalType DofOffset=0>
  class NaturalBC
/******************************************************************************/
{
// private member data
private:
    const std::string mName;
    const std::string mType;
    const std::string mSideSetName;
    Omega_h::Vector<NumDofs> mFlux;
    Plato::LinearTetCubRuleDegreeOne<SpatialDim> mCubatureRule;

// public functions
public:
    NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>(const std::string & aLoadName, Teuchos::ParameterList &aParam) :
        mName(aLoadName),
        mType(aParam.get<std::string>("Type")),
        mSideSetName(aParam.get<std::string>("Sides")),
        mCubatureRule()
    {
        auto tFlux = aParam.get<Teuchos::Array<Plato::Scalar>>("Vector");
        for(Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
        {
            mFlux(tDof) = tFlux[tDof];
        }
    }

    ~NaturalBC(){}

  /*!
    \brief Get the contribution to the assembled forcing vector.
    @param aMesh Omega_h mesh that contains sidesets.
    @param aMeshSets Omega_h mesh sets that contains sideset information.
    @param aResult Assembled vector to which the boundary terms will be added.

    The boundary terms are integrated on the parameterized surface, \f$\phi(\xi,\psi)\f$, according to:
    \f{eqnarray*}{
        \phi(\xi,\psi)=
          \left\{
           \begin{array}{ccc}
             N_I\left(\xi,\psi\right) x_I &
             N_I\left(\xi,\psi\right) y_I &
             N_I\left(\xi,\psi\right) z_I
           \end{array}
          \right\} \\
        f^{el}_{Ii} = \int_{\partial\Omega_{\xi}} N_I\left(\xi,\psi\right) t_i
              \left|\left|
                \frac{\partial\phi}{\partial\xi} \times \frac{\partial\phi}{\partial\psi}
              \right|\right| d\xi d\psi
    \f}
  */

  template<typename StateScalarType,
           typename ControlScalarType,
           typename ConfigScalarType,
           typename ResultScalarType>
  void get( Omega_h::Mesh* aMesh,
            const Omega_h::MeshSets& aMeshSets,
            Plato::ScalarMultiVectorT<  StateScalarType>& aState,
            Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
            Plato::ScalarArray3DT    < ConfigScalarType>& aConfig,
            Plato::ScalarMultiVectorT< ResultScalarType>& aResult,
            Plato::Scalar aScale) const;

  // ! Get sideset name
  decltype(mSideSetName) const& get_ss_name() const { return mSideSetName; }

  // ! Get the user-specified flux.
  decltype(mFlux) get_value() const { return mFlux; }

};


/******************************************************************************/
/*!
  \brief Owner class that contains a vector of NaturalBC objects.
*/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs=SpatialDim, Plato::OrdinalType DofsPerNode=NumDofs, Plato::OrdinalType DofOffset=0>
class NaturalBCs
/******************************************************************************/
{
private:
  std::vector<std::shared_ptr<NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>>> mBCs;

  std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>>
  setUniformNaturalBC(const std::string & aName, Teuchos::ParameterList &aSubList);

  std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>>
  setUniformComponentNaturalBC(const std::string & aName, Teuchos::ParameterList &aSubList);

public :

  /*!
    \brief Constructor that parses and creates a vector of NaturalBC objects
    based on the ParameterList.
  */
  NaturalBCs(Teuchos::ParameterList &aParams);

  /*!
    \brief Get the contribution to the assembled forcing vector from the owned boundary conditions.
    @param aMesh Omega_h mesh that contains sidesets.
    @param aMeshSets Omega_h mesh sets that contains sideset information.
    @param aResult Assembled vector to which the boundary terms will be added.
  */
  template<typename StateScalarType,
           typename ControlScalarType,
           typename ConfigScalarType,
           typename ResultScalarType>
  void get( Omega_h::Mesh* aMesh,
            const Omega_h::MeshSets& aMeshSets,
            Plato::ScalarMultiVectorT<  StateScalarType>& aState,
            Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
            Plato::ScalarArray3DT    < ConfigScalarType>& aConfig,
            Plato::ScalarMultiVectorT< ResultScalarType>& aResult,
            Plato::Scalar aScale = 1.0) const;
};

/**************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>::get
(Omega_h::Mesh* aMesh,
 const Omega_h::MeshSets& aMeshSets,
 Plato::ScalarMultiVectorT<  StateScalarType>& aState,
 Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
 Plato::ScalarArray3DT    < ConfigScalarType>& aConfig,
 Plato::ScalarMultiVectorT< ResultScalarType>& aResult,
 Plato::Scalar aScale) const
{
    // get sideset faces
    auto tFaceLids = Plato::get_face_local_ordinals(aMeshSets, this->mSideSetName);
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

template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>>
NaturalBCs<SpatialDim, NumDofs, DofsPerNode, DofOffset>::setUniformNaturalBC
(const std::string & aName, Teuchos::ParameterList &aSubList)
{
    bool tBC_Value = aSubList.isType<Plato::Scalar>("Value");
    bool tBC_Values = aSubList.isType<Teuchos::Array<Plato::Scalar>>("Values");

    const std::string tType = aSubList.get < std::string > ("Type");
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

/****************************************************************************/

/****************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
NaturalBCs<SpatialDim, NumDofs, DofsPerNode, DofOffset>::NaturalBCs(Teuchos::ParameterList &aParams) :
mBCs()
/****************************************************************************/
{
    for (Teuchos::ParameterList::ConstIterator tItr = aParams.begin(); tItr != aParams.end(); ++tItr)
    {
        const Teuchos::ParameterEntry &tEntry = aParams.entry(tItr);
        if (!tEntry.isList())
        {
            THROWERR("Parameter in Boundary Conditions block not valid.  Expect lists only.")
        }

        const std::string &tName = aParams.name(tItr);
        Teuchos::ParameterList &tSubList = aParams.sublist(tName);
        if(tSubList.isParameter("Type") == false)
        {
            std::stringstream tMsg;
            tMsg << "Natural Boundary Condition: 'Type' Parameter Keyword in Parameter Sublist: '"
                << tName.c_str() << "' is NOT defined.";
            THROWERR(tMsg.str().c_str())
        }

        const std::string tType = tSubList.get<std::string>("Type");
        std::shared_ptr<NaturalBC<SpatialDim, NumDofs, DofsPerNode, DofOffset>> tBC;
        if ("Uniform" == tType)
        {
            tBC = this->setUniformNaturalBC(tName, tSubList);
        }
        else if ("Uniform Component" == tType)
        {
            tBC = this->setUniformComponentNaturalBC(tName, tSubList);
        }
        else
        {
            std::stringstream tMsg;
            tMsg << "Natural Boundary Condition Type '" << tType.c_str() << "' is NOT supported.";
            THROWERR(tMsg.str().c_str())
        }
        mBCs.push_back(tBC);
    }
}

/**************************************************************************/
/*!
  \brief Add the boundary load to the result workset
*/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void NaturalBCs<SpatialDim,NumDofs,DofsPerNode,DofOffset>::get
(Omega_h::Mesh* aMesh,
 const Omega_h::MeshSets& aMeshSets,
 Plato::ScalarMultiVectorT<  StateScalarType>& aState,
 Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
 Plato::ScalarArray3DT    < ConfigScalarType>& aConfig,
 Plato::ScalarMultiVectorT< ResultScalarType>& aResult,
 Plato::Scalar aScale) const
{
    for (const auto &tMyBC : mBCs)
    {
        tMyBC->get(aMesh, aMeshSets, aState, aControl, aConfig, aResult, aScale);
    }
}


} // end namespace Plato 

#endif

