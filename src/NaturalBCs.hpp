#pragma once

#include <vector>
#include <memory>

#include "NaturalBC.hpp"

namespace Plato 
{

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
    std::vector<std::shared_ptr<Plato::NaturalBC<SpatialDim,NumDofs,DofsPerNode,DofOffset>>> mBCs;

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
     * \param [in]  aState        2-D view of state variables.
     * \param [in]  aControl      2-D view of control variables.
     * \param [in]  aConfig       3-D view of configuration variables.
     * \param [out] aResult       Assembled vector to which the boundary terms will be added
     * \param [in]  aScale        scalar multiplier
     *
    *******************************************************************************/
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void get(
        const Plato::SpatialModel &,
        const Plato::ScalarMultiVectorT<  StateScalarType> &,
        const Plato::ScalarMultiVectorT<ControlScalarType> &,
        const Plato::ScalarArray3DT    < ConfigScalarType> &,
        const Plato::ScalarMultiVectorT< ResultScalarType> &,
              Plato::Scalar aScale = 1.0,
              Plato::Scalar aCurrentTime = 0.0) const;
};
// class NaturalBCs

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
    bool tBC_Value = (aSubList.isType<Plato::Scalar>("Value") || aSubList.isType<std::string>("Value"));

    bool tBC_Values = (aSubList.isType<Teuchos::Array<Plato::Scalar>>("Values") ||
                       aSubList.isType<Teuchos::Array<std::string>>("Values"));

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
        if(aSubList.isType<Teuchos::Array<Plato::Scalar>>("Values"))
        {
            auto tValues = aSubList.get<Teuchos::Array<Plato::Scalar>>("Values");
            aSubList.set("Vector", tValues);
        } else
        if(aSubList.isType<Teuchos::Array<std::string>>("Values"))
        {
            auto tValues = aSubList.get<Teuchos::Array<std::string>>("Values");
            aSubList.set("Vector", tValues);
        } else
        {
            std::stringstream tMsg;
            tMsg << "Natural Boundary Condition: unexpected type encountered for 'Values' Parameter Keyword."
                 << "Specify 'type' of 'Array(double)' or 'Array(string)'.";
            THROWERR(tMsg.str().c_str())
        }
    }
    else if (tBC_Value)
    {

        auto tDof = aSubList.get<Plato::OrdinalType>("Index", 0);

        if(aSubList.isType<Plato::Scalar>("Value"))
        {
            Teuchos::Array<Plato::Scalar> tFluxVector(NumDofs, 0.0);
            auto tValue = aSubList.get<Plato::Scalar>("Value");
            tFluxVector[tDof] = tValue;
            aSubList.set("Vector", tFluxVector);
        } else
        if(aSubList.isType<std::string>("Value"))
        {
            Teuchos::Array<std::string> tFluxVector(NumDofs, "0.0");
            auto tValue = aSubList.get<std::string>("Value");
            tFluxVector[tDof] = tValue;
            aSubList.set("Vector", tFluxVector);
        } else
        {
            std::stringstream tMsg;
            tMsg << "Natural Boundary Condition: unexpected type encountered for 'Value' Parameter Keyword."
                 << "Specify 'type' of 'double' or 'string'.";
            THROWERR(tMsg.str().c_str())
        }
    }
    else
    {
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition: Uniform Boundary Condition in Parameter Sublist: '"
            << aName.c_str() << "' was NOT parsed. Check input Parameter Keywords.";
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
            THROWERR("Natural Boundary Condition: Parameter in Boundary Conditions block not valid.  Expect lists only.")
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
void NaturalBCs<SpatialDim,NumDofs,DofsPerNode,DofOffset>::get(
     const Plato::SpatialModel                           & aSpatialModel,
     const Plato::ScalarMultiVectorT <  StateScalarType> & aState,
     const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
     const Plato::ScalarArray3DT     < ConfigScalarType> & aConfig,
     const Plato::ScalarMultiVectorT < ResultScalarType> & aResult,
           Plato::Scalar aScale,
           Plato::Scalar aCurrentTime
) const
{
    for (const auto &tMyBC : mBCs)
    {
        tMyBC->get(aSpatialModel, aState, aControl, aConfig, aResult, aScale, aCurrentTime);
    }
}

}
// namespace Plato

