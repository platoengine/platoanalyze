#pragma once

#include <sstream>

#include "AnalyzeMacros.hpp"
#include "ImplicitFunctors.hpp"
#include "SurfaceLoadIntegral.hpp"
#include "SurfacePressureIntegral.hpp"

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

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aLoadName user-defined name for natural boundary condition sublist
     * \param [in] aSubList  natural boundary condition input parameter sublist
    *******************************************************************************/
    NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>(const std::string & aLoadName, Teuchos::ParameterList &aSubList) :
        mName(aLoadName),
        mType(aSubList.get<std::string>("Type")),
        mSideSetName(aSubList.get<std::string>("Sides"))
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
            Plato::SurfacePressureIntegral<SpatialDim, NumDofs, DofsPerNode, DofOffset> tSurfacePress(mSideSetName, mFlux);
            tSurfacePress(aMesh, aMeshSets, aState, aControl, aConfig, aResult, aScale);
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

