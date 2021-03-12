/*
 * StructuralDynamicsProblem.hpp
 *
 *  Created on: Apr 20, 2018
 */

#ifndef STRUCTURALDYNAMICSPROBLEM_HPP_
#define STRUCTURALDYNAMICSPROBLEM_HPP_

#include <memory>
#include <vector>
#include <sstream>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include "BLAS1.hpp"
#include "EssentialBCs.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyConstraints.hpp"

#include "geometric/GeometryScalarFunction.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/VectorFunction.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoAbstractProblem.hpp"
#include "SimplexStructuralDynamics.hpp"

#ifdef HAVE_AMGX
#include "alg/AmgXSparseLinearProblem.hpp"
#endif

namespace Plato
{

template<typename SimplexPhysics>
class StructuralDynamicsProblem: public AbstractProblem
{
private:
    static constexpr Plato::OrdinalType mSpatialDim = SimplexPhysics::mNumSpatialDims;
    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;

    Plato::OrdinalType mNumStates;
    Plato::OrdinalType mNumConfig;
    Plato::OrdinalType mNumControls;
    Plato::OrdinalType mNumIterationsAmgX;

    Plato::LocalOrdinalVector mBcDofs;

    Plato::ScalarVector mBcValues;
    Plato::ScalarVector mResidual;
    Plato::ScalarMultiVector mMyAdjoint;
    Plato::ScalarVector mGradState;
    Plato::ScalarVector mGradConfig;
    Plato::ScalarVector mGradControl;
    Plato::ScalarVector mExternalForce;

    Plato::ScalarMultiVector mGlobalState;

    std::vector<Plato::Scalar> mFreqArray;
    Teuchos::RCP<Plato::CrsMatrixType> mJacobian;

    // required
    std::shared_ptr<const Plato::Elliptic::VectorFunction<SimplexPhysics>> mEquality;

    // optional
    std::shared_ptr<const Plato::Elliptic::PhysicsScalarFunction<SimplexPhysics>> mObjective;
    std::shared_ptr<const Plato::Geometric::GeometryScalarFunction<SimplexPhysics>> mConstraint;
    std::shared_ptr<const Plato::Elliptic::VectorFunction<SimplexPhysics>> mAdjointProb;

public:
    /******************************************************************************//**
     *
     * @brief Constructor
     * @param aMesh mesh data base
     * @param aMeshSets mesh sets data base
     * @param aParamList parameter list with input data
     *
    **********************************************************************************/
    StructuralDynamicsProblem(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList & aParamList) :
            mNumStates(aMesh.nverts() * mNumDofsPerNode),
            mNumConfig(aMesh.nverts() * mSpatialDim),
            mNumControls(aMesh.nverts()),
            mNumIterationsAmgX(1000),
            mResidual("Residual", mNumStates),
            mGradState("GradState", mNumStates),
            mGradConfig("GradConfig", mNumConfig),
            mGradControl("GradControl", mNumControls),
            mExternalForce("BoundaryLoads", mNumStates),
            mFreqArray(),
            mJacobian(Teuchos::null),
            mEquality(nullptr),
            mObjective(nullptr),
            mConstraint(nullptr),
            mAdjointProb(nullptr)
    {
        this->initialize(aMesh, aMeshSets, aParamList);
        this->readFrequencyArray(aParamList);
    }

    /******************************************************************************//**
     *
     * @brief Return number of degrees of freedom in solution.
     * @return Number of degrees of freedom
     *
    **********************************************************************************/
    Plato::OrdinalType getNumSolutionDofs()
    {
        return SimplexPhysics::mNumDofsPerNode;
    }

    /******************************************************************************//**
     *
     * @brief Constructor
     * @param aMesh mesh data base
     * @param aEquality equality constraint vector function
     *
    **********************************************************************************/
    StructuralDynamicsProblem(Omega_h::Mesh& aMesh, std::shared_ptr<Plato::Elliptic::VectorFunction<SimplexPhysics>> & aEquality) :
            mNumStates(aMesh.nverts() * mNumDofsPerNode),
            mNumConfig(aMesh.nverts() * mSpatialDim),
            mNumControls(aMesh.nverts()),
            mNumIterationsAmgX(1000),
            mResidual("Residual", mNumStates),
            mGradState("GradState", mNumStates),
            mGradConfig("GradConfig", mNumConfig),
            mGradControl("GradControl", mNumControls),
            mExternalForce("ExternalForce", mNumStates),
            mFreqArray(),
            mJacobian(Teuchos::null),
            mEquality(aEquality),
            mObjective(nullptr),
            mConstraint(nullptr),
            mAdjointProb(nullptr)
    {
    }

    /******************************************************************************//**
     *
     * @brief Destructor
     *
    **********************************************************************************/
    virtual ~StructuralDynamicsProblem()
    {
    }

    /******************************************************************************//**
     *
     * @brief Set array of angular frequencies and allocate state container
     *
     * @param[in] aInput angular frequencies
     *
    **********************************************************************************/
    void setFrequencyArray(const std::vector<Plato::Scalar>& aInput)
    {
        assert(aInput.size() > static_cast<size_t>(0));
        mFreqArray = aInput;
        mGlobalState = Plato::ScalarMultiVector("States", mFreqArray.size(), mNumStates);
    }

    /******************************************************************************//**
     *
     * @brief Set essential boundary conditions
     *
     * @param[in] aBcDofs degrees of freedom associated with essential boundary conditions
     * @param[in] aBcValues values associated with essential boundary conditions
     *
    **********************************************************************************/
    void setEssentialBoundaryConditions(const Plato::LocalOrdinalVector & aBcDofs,
                                        const Plato::ScalarVector & aBcValues)
    {
        assert(aBcDofs.size() > 0);
        assert(aBcValues.size() > 0);
        Kokkos::resize(mBcDofs, aBcDofs.size());
        Kokkos::deep_copy(mBcDofs, aBcDofs);
        Kokkos::resize(mBcValues, aBcValues.size());
        Kokkos::deep_copy(mBcValues, aBcValues);
    }

    /******************************************************************************//**
     *
     * @brief Set external force vector
     *
     * @param[in] aInput external force vector
     *
    **********************************************************************************/
    void setExternalForce(const Plato::ScalarVector & aInput)
    {
        assert(static_cast<Plato::OrdinalType>(aInput.size()) == mNumStates);
        assert(static_cast<Plato::OrdinalType>(mExternalForce.size()) == mNumStates);
        Kokkos::deep_copy(mExternalForce, aInput);
    }

    /******************************************************************************//**
    * 
    * @brief Set maximum number of AmgX solver iterations.
    * @ param [in] aInput number of iterations 
    *
    **********************************************************************************/
    void setMaxNumIterationsAmgX(const Plato::OrdinalType& aInput)
    {
        mNumIterationsAmgX = aInput;   
    }

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aGlobalState 2D container of state variables
     * @param [in] aControl 1D container of control variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::Solution & aGlobalSolution) override
    { return; }

    /******************************************************************************//**
     *
     * @brief Set state vector
     *
     * @param[in] aInput state vector
     *
    **********************************************************************************/
    void setGlobalSolution(const Plato::Solution & aSolution) override
    {
        assert(aSolution.State.size() == mGlobalState.size());
        Kokkos::deep_copy(mGlobalState, aSolution.State);
    }

    /******************************************************************************//**
     *
     * @brief Get state vector
     *
    **********************************************************************************/
    Plato::Solution getGlobalSolution() override
    {
        return Plato::Solution(mGlobalState);
    }

    /******************************************************************************//**
     *
     * @brief Get adjoint vector for last time step
     *
    **********************************************************************************/
    Plato::Adjoint getAdjoint()
    {
        return Plato::Adjoint(mMyAdjoint); /* Returns adjoint solution for last time step. */
    }

    /******************************************************************************/
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)
    /******************************************************************************/
    {
        if(mJacobian->isBlockMatrix())
        {
            Plato::applyBlockConstraints<mNumDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues);
        }
        else
        {
            Plato::applyConstraints<mNumDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues);
        }
    }

    /******************************************************************************/
    void applyBoundaryLoads(const Plato::ScalarVector & aForce)
    /******************************************************************************/
    {
        auto tBoundaryLoads = mExternalForce;
        auto tTotalNumDofs = aForce.size();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tTotalNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aDofOrdinal){
            aForce(aDofOrdinal) += tBoundaryLoads(aDofOrdinal);
        }, "add boundary loads");
    }

    /******************************************************************************/
    Plato::Solution solution(const Plato::ScalarVector & aControl)
    /******************************************************************************/
    {
        assert(aControl.size() == mNumControls);

        const Plato::OrdinalType tNumFreqs = mFreqArray.size();
        for(Plato::OrdinalType tFreqIndex = 0; tFreqIndex < tNumFreqs; tFreqIndex++)
        {
            assert(mResidual.size() == mNumStates);
            auto tMyStatesSubView = Kokkos::subview(mGlobalState, tFreqIndex, Kokkos::ALL());
            assert(tMyStatesSubView.size() == mNumStates);
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tMyStatesSubView);
            auto tMyFrequency = mFreqArray[tFreqIndex];
            mResidual = mEquality->value(tMyStatesSubView, aControl, tMyFrequency);
            this->applyBoundaryLoads(mResidual);

            mJacobian = mEquality->gradient_u(tMyStatesSubView, aControl, tMyFrequency);
            this->applyConstraints(mJacobian, mResidual);

#ifdef HAVE_AMGX
            using AmgXLinearProblem = Plato::AmgXSparseLinearProblem<Plato::OrdinalType, mNumDofsPerNode>;
            auto tConfigString = Plato::get_config_string();
            auto tSolver = std::make_shared<AmgXLinearProblem>(*mJacobian, tMyStatesSubView, mResidual, tConfigString);
            tSolver->solve();
#endif
        }

        return Plato::Solution(mGlobalState);
    }

    /******************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControl) override
    /******************************************************************************/
    {
        assert(aControl.size() == mNumControls);

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << "\nMESSAGE: OBJECTIVE VALUE REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER.\n"
                    << "USER SHOULD MAKE SURE THAT THE OBJECTIVE FUNCTION IS DEFINED IN INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        Plato::Scalar tValue = mObjective->value(Plato::Solution(mGlobalState), aControl);

        return tValue;
    }

    /******************************************************************************/
    Plato::Scalar
    objectiveValue(const Plato::ScalarVector & aControl, const Plato::Solution & aGlobalSolution) override
    /******************************************************************************/
    {
        auto tGlobalState = aGlobalSolution.State;
        assert(aControl.size() == mNumControls);
        assert(tGlobalState.extent(0) == mGlobalState.extent(0));
        assert(tGlobalState.extent(1) == mGlobalState.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << "\nMESSAGE: OBJECTIVE VALUE REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER.\n"
                    << "USER SHOULD MAKE SURE THAT THE OBJECTIVE FUNCTION IS DEFINED IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        auto tValue = mObjective->value(Plato::Solution(aGlobalSolution), aControl);

        return tValue;
    }

    /******************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControl) override
    /******************************************************************************/
    {
        assert(aControl.size() == mNumControls);

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << "\nMESSAGE: CONSTRAINT VALUE REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER.\n"
                    << "USER SHOULD MAKE SURE THAT THE CONSTRAINT FUNCTION IS DEFINED IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        auto tValue = mConstraint->value(aControl);

        return tValue;
    }

    /******************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl) override
    /******************************************************************************/
    {
        assert(aControl.size() == mNumControls);

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << "\nMESSAGE: OBJECTIVE GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER.\n"
                    << "USER SHOULD MAKE SURE THAT THE OBJECTIVE FUNCTION IS DEFINED IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        return mObjective->gradient_z(Plato::Solution(mGlobalState), aControl);
    }

    /******************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControl, const Plato::Solution & aGlobalSolution)
    /******************************************************************************/
    {
        assert(aControl.size() == mNumControls);

        auto tGlobalState = aGlobalSolution.State;
        assert(tGlobalState.extent(0) == mGlobalState.extent(0));
        assert(tGlobalState.extent(1) == mGlobalState.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << "\nMESSAGE: OBJECTIVE GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER.\n"
                    << "USER SHOULD MAKE SURE THAT THE OBJECTIVE FUNCTION IS DEFINED IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        // compute dfdu: partial of objective wrt u and dfdz: partial of objective wrt z
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mGradState);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mGradControl);
        const Plato::OrdinalType tNumFreqs = mFreqArray.size();
        for(Plato::OrdinalType tFreqIndex = 0; tFreqIndex < tNumFreqs; tFreqIndex++)
        {
            auto tMyFrequency = mFreqArray[tFreqIndex];
            auto tMyStatesSubView = Kokkos::subview(tGlobalState, tFreqIndex, Kokkos::ALL());
            assert(tMyStatesSubView.size() == mNumStates);

            auto tPartialObjectiveWrtState = mObjective->gradient_u(aGlobalSolution, aControl, tFreqIndex, tMyFrequency);
            Plato::blas1::update(static_cast<Plato::Scalar>(1.0), tPartialObjectiveWrtState, static_cast<Plato::Scalar>(1.0), mGradState);

            auto tPartialObjectiveWrtControl = mObjective->gradient_z(aGlobalSolution, aControl, tMyFrequency);
            Plato::blas1::update(static_cast<Plato::Scalar>(1.0), tPartialObjectiveWrtControl, static_cast<Plato::Scalar>(1.0), mGradControl);
        }

        Plato::blas1::scale(static_cast<Plato::Scalar>(-1), mGradState);
        this->addResidualContribution(Plato::partial::CONTROL, aControl, tGlobalState, mGradControl);

        return mGradControl;
    }

    /******************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControl) override
    /******************************************************************************/
    {
        assert(aControl.size() == mNumControls);

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << "\nMESSAGE: OBJECTIVE CONFIGURATION GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER.\n"
                    << "USER SHOULD MAKE SURE THAT THE OBJECTIVE FUNCTION IS DEFINED IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        return mObjective->gradient_x(Plato::Solution(mGlobalState), aControl);
    }

    /******************************************************************************/
    Plato::ScalarVector
    objectiveGradientX(const Plato::ScalarVector & aControl, const Plato::Solution & aGlobalSolution)
    /******************************************************************************/
    {
        assert(aControl.size() == mNumControls);

        auto tGlobalState = aGlobalSolution.State;
        assert(tGlobalState.extent(0) == mGlobalState.extent(0));
        assert(tGlobalState.extent(1) == mGlobalState.extent(1));

        if(mObjective == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << "\nMESSAGE: OBJECTIVE CONFIGURATION GRADIENT REQUESTED BUT OBJECTIVE PTR WAS NOT DEFINED BY THE USER.\n"
                    << "USER SHOULD MAKE SURE THAT THE OBJECTIVE FUNCTION IS DEFINED IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mGradState);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mGradConfig);
        const Plato::OrdinalType tNumFreqs = mFreqArray.size();
        for(Plato::OrdinalType tFreqIndex = 0; tFreqIndex < tNumFreqs; tFreqIndex++)
        {
            auto tMyFrequency = mFreqArray[tFreqIndex];

            // Compute partial derivative of objective function wrt configuration for this time step
            auto tMyStatesSubView = Kokkos::subview(tGlobalState, tFreqIndex, Kokkos::ALL());
            assert(tMyStatesSubView.size() == mNumStates);
            auto tPartialObjectiveWrtConfig = mObjective->gradient_x(aGlobalSolution, aControl, tMyFrequency);
            Plato::blas1::update(static_cast<Plato::Scalar>(1.0), tPartialObjectiveWrtConfig, static_cast<Plato::Scalar>(1.0), mGradConfig);

            // Compute partial derivative of objective function wrt state for this time step
            auto tPartialObjectiveWrtState = mObjective->gradient_u(aGlobalSolution, aControl, tFreqIndex, tMyFrequency);
            Plato::blas1::update(static_cast<Plato::Scalar>(1.0), tPartialObjectiveWrtState, static_cast<Plato::Scalar>(1.0), mGradState);
        }

        Plato::blas1::scale(static_cast<Plato::Scalar>(-1), mGradState);
        this->addResidualContribution(Plato::partial::CONFIGURATION, aControl, tGlobalState, mGradConfig);

        return mGradConfig;
    }

    /******************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControl) override
    /******************************************************************************/
    {
        assert(aControl.size() == mNumControls);

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << "\nMESSAGE: CONSTRAINT GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER.\n"
                    << "USER SHOULD MAKE SURE THAT THE CONSTRAINT FUNCTION IS DEFINED IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        return mConstraint->gradient_z(aControl);
    }

    /******************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControl) override
    /******************************************************************************/
    {
        assert(aControl.size() == mNumControls);

        if(mConstraint == nullptr)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__
                    << "\nMESSAGE: CONSTRAINT CONFIGURATION GRADIENT REQUESTED BUT CONSTRAINT PTR WAS NOT DEFINED BY THE USER.\n"
                    << "USER SHOULD MAKE SURE THAT THE CONSTRAINT FUNCTION IS DEFINED IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }

        return mConstraint->gradient_x(aControl);
    }

private:
    /******************************************************************************/
    void initialize(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aParamList)
    /******************************************************************************/
    {
        auto tEqualityName = aParamList.get<std::string>("PDE Constraint");
        mEquality = std::make_shared<Plato::Elliptic::VectorFunction<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aParamList, tEqualityName);

        if(aParamList.isType<std::string>("Constraint"))
        {
            std::string tConstraintName = aParamList.get<std::string>("Constraint");
            mConstraint = std::make_shared<Plato::Geometric::GeometryScalarFunction<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aParamList, tConstraintName);
        }

        if(aParamList.isType<std::string>("Objective"))
        {
            std::string tObjectiveName = aParamList.get<std::string>("Objective");
            mObjective = std::make_shared<Plato::Elliptic::PhysicsScalarFunction<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aParamList, tObjectiveName);

            auto tLength = mEquality->size();
            mMyAdjoint = Plato::ScalarMultiVector("MyAdjoint", 1, tLength);

            std::string tAdjointName = "StructuralDynamics Adjoint";
            mAdjointProb = std::make_shared<Plato::Elliptic::VectorFunction<SimplexPhysics>>(aMesh, aMeshSets, mDataMap, aParamList, tAdjointName);
        }

        // Parse essential boundary conditions (i.e. Dirichlet)
        //
        Plato::EssentialBCs<SimplexPhysics>
            tEssentialBoundaryConditions(aParamList.sublist("Essential Boundary Conditions",false), aMeshSets);
        tEssentialBoundaryConditions.get(mBcDofs, mBcValues);
    }

    /******************************************************************************/
    void readFrequencyArray(Teuchos::ParameterList& aParamList)
    /******************************************************************************/
    {
        if(aParamList.isSublist("Frequency Steps") == true)
        {
            auto tFreqParams = aParamList.sublist("Frequency Steps");
            assert(tFreqParams.isParameter("Values"));
            auto tFreqValues = tFreqParams.get < Teuchos::Array < Plato::Scalar >> ("Values");

            const Plato::OrdinalType tNumFrequencies = tFreqValues.size();
            mFreqArray.resize(tNumFrequencies);
            assert(mFreqArray.size() == static_cast<size_t>(tFreqValues.size()));

            assert(mEquality->size() == mNumStates);
            mGlobalState = Plato::ScalarMultiVector("States", tNumFrequencies, mNumStates);

            for(Plato::OrdinalType tIndex = 0; tIndex < tNumFrequencies; tIndex++)
            {
                mFreqArray[tIndex] = tFreqValues[tIndex];
            }
        }
        else
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                    << ", LINE: " << __LINE__ << "\nMESSAGE: FREQUENCY STEPS SUBLIST IS NOT DEFINED IN THE INPUT FILE.\n"
                    << "\nUSER SHOULD DEFINE FREQUENCY STEPS SUBLIST IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
    }

    /******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType> computePartialResidualWrtDesignVar(const Plato::partial::derivative_t & aWhichType,
                                                                          const Plato::ScalarVector & aGlobalState,
                                                                          const Plato::ScalarVector & aControl,
                                                                          const Plato::Scalar & aTimeStep)
    /******************************************************************************/
    {
        switch(aWhichType)
        {
            case Plato::partial::STATE:
            {
                return mEquality->gradient_u(aGlobalState, aControl, aTimeStep);
            }
            case Plato::partial::CONTROL:
            {
                return mEquality->gradient_z(aGlobalState, aControl, aTimeStep);
            }
            case Plato::partial::CONFIGURATION:
            {
                return mEquality->gradient_x(aGlobalState, aControl, aTimeStep);
            }
        }
        return (Teuchos::null);
    }

    /******************************************************************************/
    void addResidualContribution(const Plato::partial::derivative_t & aWhichPartial,
                                 const Plato::ScalarVector & aControl,
                                 const Plato::ScalarMultiVector & aGlobalState,
                                 Plato::ScalarVector & aOutput)
    /******************************************************************************/
    {
        const Plato::OrdinalType tNumFreqs = mFreqArray.size();
        for(Plato::OrdinalType tFreqIndex = 0; tFreqIndex < tNumFreqs; tFreqIndex++)
        {
            // compute dgdu: partial of PDE wrt state
            auto tMyFrequency = mFreqArray[tFreqIndex];
            auto tMyStatesSubView = Kokkos::subview(aGlobalState, tFreqIndex, Kokkos::ALL());
            mJacobian = mAdjointProb->gradient_u(tMyStatesSubView, aControl, tMyFrequency);
            this->applyConstraints(mJacobian, mGradState);

            // adjoint problem \lambda = (dg/du)-*(df/du) uses transpose of global stiffness,
            const Plato::OrdinalType tTIME_STEP_INDEX = 0;
            Plato::ScalarVector tAdjoint = Kokkos::subview(mMyAdjoint, tTIME_STEP_INDEX, Kokkos::ALL());
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tAdjoint);
#ifdef HAVE_AMGX
            using AmgXLinearProblem = Plato::AmgXSparseLinearProblem< Plato::OrdinalType, mNumDofsPerNode>;
            auto tConfigString = Plato::get_config_string();
            auto tSolver = std::make_shared<AmgXLinearProblem>(*mJacobian, tAdjoint, mGradState, tConfigString);
            tSolver->solve();
#endif

            // compute dgdz: partial of PDE wrt design variable.
            auto tPartialWrtDesignVar =
                    this->computePartialResidualWrtDesignVar(aWhichPartial, tMyStatesSubView, aControl, tMyFrequency);

            // compute dfdz + dgdz . adjoint
            Plato::MatrixTimesVectorPlusVector(tPartialWrtDesignVar, tAdjoint, aOutput);
        }
    }
};
// class StructuralDynamicsProblem

}// namespace Plato

#endif /* STRUCTURALDYNAMICSPROBLEM_HPP_ */
