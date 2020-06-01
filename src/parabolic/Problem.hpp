#pragma once

#include "BLAS1.hpp"
#include "EssentialBCs.hpp"
#include "AnalyzeMacros.hpp"
#include "SimplexMechanics.hpp"
#include "ApplyConstraints.hpp"
#include "PlatoAbstractProblem.hpp"
#include "alg/PlatoSolverFactory.hpp"

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Geometrical.hpp"
#include "ComputedField.hpp"
#include "parabolic/TrapezoidIntegrator.hpp"
#include "parabolic/VectorFunction.hpp"
#include "parabolic/ScalarFunctionBase.hpp"
#include "parabolic/ScalarFunctionBaseFactory.hpp"
#include "elliptic/ScalarFunctionBase.hpp"
#include "elliptic/ScalarFunctionBaseFactory.hpp"
#include "geometric/ScalarFunctionBase.hpp"
#include "geometric/ScalarFunctionBaseFactory.hpp"

namespace Plato
{

namespace Parabolic
{

    template<typename SimplexPhysics>
    class Problem: public Plato::AbstractProblem
    {
      private:
        static constexpr Plato::OrdinalType SpatialDim = SimplexPhysics::mNumSpatialDims;
        static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;

        Plato::Parabolic::VectorFunction<SimplexPhysics> mPDEConstraint;

        Plato::Parabolic::TrapezoidIntegrator<SimplexPhysics> mTrapezoidIntegrator;

        Plato::OrdinalType mNumSteps;
        Plato::Scalar      mTimeStep;

        bool mSaveState;

        std::shared_ptr<const Plato::Geometric::ScalarFunctionBase> mConstraint;
        std::shared_ptr<const Plato::Parabolic::ScalarFunctionBase> mObjective;

        Plato::ScalarVector mResidual;
        Plato::ScalarVector mResidualV;

        Plato::ScalarMultiVector mAdjoints_U;
        Plato::ScalarMultiVector mAdjoints_V;

        Plato::ScalarMultiVector mState;
        Plato::ScalarMultiVector mStateDot;

        Teuchos::RCP<Plato::CrsMatrixType> mJacobianU;
        Teuchos::RCP<Plato::CrsMatrixType> mJacobianV;

        Teuchos::RCP<Plato::ComputedFields<SpatialDim>> mComputedFields;

        Plato::LocalOrdinalVector mStateBcDofs;
        Plato::ScalarVector mStateBcValues;

        rcp<Plato::AbstractSolver> mSolver;
      public:
        /******************************************************************************/
        Problem(
          Omega_h::Mesh& aMesh,
          Omega_h::MeshSets& aMeshSets,
          Teuchos::ParameterList& aParamList,
          Comm::Machine aMachine
        ) :
            mPDEConstraint   (aMesh, aMeshSets, mDataMap, aParamList, 
                                   aParamList.get<std::string>("PDE Constraint")),
            mTrapezoidIntegrator    (aParamList.sublist("Time Integration")),
            mNumSteps     (aParamList.sublist("Time Integration").get<int>("Number Time Steps")),
            mTimeStep     (aParamList.sublist("Time Integration").get<Plato::Scalar>("Time Step")),
            mSaveState    (aParamList.sublist("Parabolic").isType<Teuchos::Array<std::string>>("Plottable")),
            mObjective    (nullptr),
            mConstraint   (nullptr),
            mResidual     ("MyResidual", mPDEConstraint.size()),
            mState        ("State",      mNumSteps, mPDEConstraint.size()),
            mStateDot     ("StateDot",   mNumSteps, mPDEConstraint.size()),
            mJacobianU(Teuchos::null),
            mJacobianV(Teuchos::null)
        /******************************************************************************/
        {
            // parse boundary constraints
            //
            Plato::EssentialBCs<SimplexPhysics>
                tEssentialBoundaryConditions(aParamList.sublist("Essential Boundary Conditions",false));
            tEssentialBoundaryConditions.get(aMeshSets, mStateBcDofs, mStateBcValues);

            // parse constraint
            //
            if(aParamList.isType<std::string>("Constraint"))
            {
                std::string tName = aParamList.get<std::string>("Constraint");
                Plato::Geometric::ScalarFunctionBaseFactory<Plato::Geometrical<SpatialDim>> tScalarFunctionBaseFactory;
                mConstraint = tScalarFunctionBaseFactory.create(aMesh, aMeshSets, mDataMap, aParamList, tName);
            }

            // parse objective
            //
            if(aParamList.isType<std::string>("Objective"))
            {
                std::string tName = aParamList.get<std::string>("Objective");
                Plato::Parabolic::ScalarFunctionBaseFactory<SimplexPhysics> tScalarFunctionBaseFactory;
                mObjective = tScalarFunctionBaseFactory.create(aMesh, aMeshSets, mDataMap, aParamList, tName);

                auto tLength = mPDEConstraint.size();
                mAdjoints_U = Plato::ScalarMultiVector("MyAdjoint U", mNumSteps, tLength);
                mAdjoints_V = Plato::ScalarMultiVector("MyAdjoint V", mNumSteps, tLength);
            }

            // parse computed fields
            //
            if(aParamList.isSublist("Computed Fields"))
            {
              mComputedFields = Teuchos::rcp(new Plato::ComputedFields<SpatialDim>(aMesh, aParamList.sublist("Computed Fields")));
            }

            // parse initial state
            //
            if(aParamList.isSublist("Initial State"))
            {
                Plato::ScalarVector tInitialState = Kokkos::subview(mState, 0, Kokkos::ALL());
                if(mComputedFields == Teuchos::null) {
                  THROWERR("No 'Computed Fields' have been defined");
                }

                auto tDofNames = mPDEConstraint.getDofNames();
    
                auto tInitStateParams = aParamList.sublist("Initial State");
                for (auto i = tInitStateParams.begin(); i != tInitStateParams.end(); ++i) {
                    const auto &tEntry = tInitStateParams.entry(i);
                    const auto &tName  = tInitStateParams.name(i);

                    if (tEntry.isList())
                    {
                        auto& tStateList = tInitStateParams.sublist(tName);
                        auto tFieldName = tStateList.get<std::string>("Computed Field");
                        int tDofIndex = -1;
                        for (int j = 0; j < tDofNames.size(); ++j)
                        {
                            if (tDofNames[j] == tName) {
                               tDofIndex = j;
                            }
                        }
                        mComputedFields->get(tFieldName, tDofIndex, tDofNames.size(), tInitialState);
                    }
                }
            }

            Plato::SolverFactory tSolverFactory(aParamList.sublist("Linear Solver"));
            mSolver = tSolverFactory.create(aMesh, aMachine, SimplexPhysics::mNumDofsPerNode);

        }
        /******************************************************************************//**
         * @brief Return number of degrees of freedom in solution.
         * @return Number of degrees of freedom
        **********************************************************************************/
        Plato::OrdinalType getNumSolutionDofs()
        {
            return SimplexPhysics::mNumDofsPerNode;
        }
        /******************************************************************************/
        Plato::Solution getGlobalSolution()
        /******************************************************************************/
        {
            return Plato::Solution(mState, mStateDot);
        }

        /******************************************************************************/
        Plato::Adjoint getAdjoint()
        /******************************************************************************/
        {
            return Plato::Adjoint(mAdjoints_U, mAdjoints_V);
        }
        /******************************************************************************/
        void setGlobalSolution(const Plato::Solution & aSolution)
        /******************************************************************************/
        {
            auto tStates = aSolution.State;
            assert(tStates.extent(0) == mState.extent(0));
            assert(tStates.extent(1) == mState.extent(1));

            Kokkos::deep_copy(mState, tStates);

            auto tStatesDot = aSolution.StateDot;
            assert(tStatesDot.extent(0) == mStateDot.extent(0));
            assert(tStatesDot.extent(1) == mStateDot.extent(1));

            Kokkos::deep_copy(mStateDot, tStatesDot);
        }
        void applyConstraints(
          const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
          const Plato::ScalarVector & aVector){}
  
        /******************************************************************************/
        void applyStateDotConstraints(
          const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
          const Plato::ScalarVector & aVector,
          Plato::Scalar aScale
        )
        /******************************************************************************/
        {
            if(mJacobianU->isBlockMatrix())
            {
                Plato::applyBlockConstraints<mNumDofsPerNode>(aMatrix, aVector, mStateBcDofs, mStateBcValues, aScale);
            }
            else
            {
                Plato::applyConstraints<mNumDofsPerNode>(aMatrix, aVector, mStateBcDofs, mStateBcValues, aScale);
            }
        }
        void applyBoundaryLoads(const Plato::ScalarVector & aForce){}
        /******************************************************************************//**
         * @brief Update physics-based parameters within optimization iterations
         * @param [in] aState 2D container of state variables
         * @param [in] aControl 1D container of control variables
        **********************************************************************************/
        void updateProblem(const Plato::ScalarVector & aControl, const Plato::Solution & aSolution)
        { return; }
        /******************************************************************************/
        Plato::Solution
        solution(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {

            mDataMap.clearStates();
            Plato::ScalarVector tStateInit    = Kokkos::subview(mState,    /*StepIndex=*/0, Kokkos::ALL());
            Plato::ScalarVector tStateDotInit = Kokkos::subview(mStateDot, /*StepIndex=*/0, Kokkos::ALL());
            mResidual  = mPDEConstraint.value(tStateInit, tStateDotInit, aControl, mTimeStep);
            mDataMap.saveState();
   
            for(Plato::OrdinalType tStepIndex = 1; tStepIndex < mNumSteps; tStepIndex++) {
              Plato::ScalarVector tStatePrev    = Kokkos::subview(mState,    tStepIndex-1, Kokkos::ALL());
              Plato::ScalarVector tStateDotPrev = Kokkos::subview(mStateDot, tStepIndex-1, Kokkos::ALL());
              Plato::ScalarVector tState        = Kokkos::subview(mState,    tStepIndex,   Kokkos::ALL());
              Plato::ScalarVector tStateDot     = Kokkos::subview(mStateDot, tStepIndex,   Kokkos::ALL());

              // -R_{u}
              mResidual  = mPDEConstraint.value(tState, tStateDot, aControl, mTimeStep);
              Plato::blas1::scale(-1.0, mResidual);

              // R_{v}
              mResidualV = mTrapezoidIntegrator.v_value(tState,    tStatePrev,
                                                        tStateDot, tStateDotPrev, mTimeStep);

              // R_{u,v^N}
              mJacobianV = mPDEConstraint.gradient_v(tState, tStateDot, aControl, mTimeStep);

              // -R_{u} += R_{u,v^N} R_{v}
              Plato::MatrixTimesVectorPlusVector(mJacobianV, mResidualV, mResidual);

              // R_{u,u^N}
              mJacobianU = mPDEConstraint.gradient_u(tState, tStateDot, aControl, mTimeStep);

              // R_{v,u^N}
              auto tR_vu = mTrapezoidIntegrator.v_grad_u(mTimeStep);

              // R_{u,u^N} += R_{u,v^N} R_{v,u^N}
              Plato::blas1::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

              this->applyStateDotConstraints(mJacobianU, mResidual, mTimeStep);

              Plato::ScalarVector tDeltaD("increment", tState.extent(0));
              Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaD);

              // compute displacement increment:
              mSolver->solve(*mJacobianU, tDeltaD, mResidual);

              // compute and add statedot increment: \Delta v = - ( R_{v} + R_{v,u} \Delta u )
              Plato::blas1::axpy(tR_vu, tDeltaD, mResidualV);
              // a_{k+1} = a_{k} + \Delta a
              Plato::blas1::axpy(-1.0, mResidualV, tStateDot);

              // add displacement increment
              Plato::blas1::axpy(1.0, tDeltaD, tState);

              if ( mSaveState )
              {
                // evaluate at new state
                mResidual  = mPDEConstraint.value(tState, tStateDot, aControl, mTimeStep);
                mDataMap.saveState();
              }
            }
            return Plato::Solution(mState, mStateDot);
        }

        /******************************************************************************/
        Plato::Scalar constraintValue(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            if(mConstraint == nullptr)
            {
                THROWERR("CONSTRAINT REQUESTED BUT NOT DEFINED BY USER.");
            }
            return mConstraint->value(aControl);
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradient(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            if(mConstraint == nullptr)
            {
                THROWERR("CONSTRAINT REQUESTED BUT NOT DEFINED BY USER.");
            }
            return mConstraint->gradient_z(aControl);
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradientX(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            if(mConstraint == nullptr)
            {
                THROWERR("CONSTRAINT REQUESTED BUT NOT DEFINED BY USER.");
            }
            return mConstraint->gradient_x(aControl);
        }
        /******************************************************************************/
        Plato::Scalar objectiveValue(
          const Plato::ScalarVector & aControl,
          const Plato::Solution & aSolution
        )
        /******************************************************************************/
        {
            if(mObjective == nullptr)
            {
                THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
            }
            auto tSolution = solution(aControl);
            return mObjective->value(tSolution, aControl, mTimeStep);
        }
        /******************************************************************************/
        Plato::Scalar objectiveValue(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            if(mObjective == nullptr)
            {
                THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
            }
            return mObjective->value(Plato::Solution(mState, mStateDot), aControl, mTimeStep);
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradient(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            if(mObjective == nullptr)
            {
                THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
            }
            auto tSolution = solution(aControl);
            return objectiveGradient(aControl, tSolution);
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradient(
          const Plato::ScalarVector & aControl,
          const Plato::Solution & aSolution
        )
        /******************************************************************************/
        {
            if(mObjective == nullptr)
            {
                THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
            }

            auto tSolution = Plato::Solution(mState, mStateDot);

            // F_{,z}
            auto t_dFdz = mObjective->gradient_z(tSolution, aControl, mTimeStep);

            auto tLastStepIndex = mNumSteps - 1;
            for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--) {

                auto tU = Kokkos::subview(mState, tStepIndex, Kokkos::ALL());
                auto tV = Kokkos::subview(mStateDot, tStepIndex, Kokkos::ALL());

                Plato::ScalarVector tAdjoint_U = Kokkos::subview(mAdjoints_U, tStepIndex, Kokkos::ALL());
                Plato::ScalarVector tAdjoint_V = Kokkos::subview(mAdjoints_V, tStepIndex, Kokkos::ALL());

                // F_{,u^k}
                auto t_dFdu = mObjective->gradient_u(tSolution, aControl, tStepIndex, mTimeStep);
                // F_{,v^k}
                auto t_dFdv = mObjective->gradient_v(tSolution, aControl, tStepIndex, mTimeStep);

                if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1

                    // L_{v}^{k+1}
                    Plato::ScalarVector tAdjoint_V_next = Kokkos::subview(mAdjoints_V, tStepIndex+1, Kokkos::ALL());

                    // R_{v,u^k}^{k+1}
                    auto tR_vu_prev = mTrapezoidIntegrator.v_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{v}^{k+1} R_{v,u^k}^{k+1}
                    Plato::blas1::axpy(tR_vu_prev, tAdjoint_V_next, t_dFdu);


                    // R_{v,v^k}^{k+1}
                    auto tR_vv_prev = mTrapezoidIntegrator.v_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{v}^{k+1} R_{v,v^k}^{k+1}
                    Plato::blas1::axpy(tR_vv_prev, tAdjoint_V_next, t_dFdv);

                }
                Plato::blas1::scale(static_cast<Plato::Scalar>(-1), t_dFdu);

                // R_{v,u^k}
                auto tR_vu = mTrapezoidIntegrator.v_grad_u(mTimeStep);

                // -F_{,u^k} += R_{v,u^k}^k F_{,v^k}
                Plato::blas1::axpy(tR_vu, t_dFdv, t_dFdu);

                // R_{u,u^k}
                mJacobianU = mPDEConstraint.gradient_u(tU, tV, aControl, mTimeStep);

                // R_{u,v^k}
                mJacobianV = mPDEConstraint.gradient_v(tU, tV, aControl, mTimeStep);

                // R_{u,u^k} -= R_{v,u^k} R_{u,v^k}
                Plato::blas1::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

                this->applyConstraints(mJacobianU, t_dFdu);

                // L_u^k
                mSolver->solve(*mJacobianU, tAdjoint_U, t_dFdu);

                // L_v^k
                Plato::MatrixTimesVectorPlusVector(mJacobianV, tAdjoint_U, t_dFdv);
                Plato::blas1::fill(0.0, tAdjoint_V);
                Plato::blas1::axpy(-1.0, t_dFdv, tAdjoint_V);

                // R^k_{,z}
                auto t_dRdz = mPDEConstraint.gradient_z(tU, tV, aControl, mTimeStep);

                // F_{,z} += L_u^k R^k_{,z}
                Plato::MatrixTimesVectorPlusVector(t_dRdz, tAdjoint_U, t_dFdz);
            }

            return t_dFdz;
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradientX(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            if(mObjective == nullptr)
            {
                THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
            }
            auto tSolution = solution(aControl);
            return objectiveGradientX(aControl, tSolution);
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradientX(
          const Plato::ScalarVector & aControl,
          const Plato::Solution & aSolution
        )
        /******************************************************************************/
        {
            if(mObjective == nullptr)
            {
                THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
            }

            auto tSolution = Plato::Solution(mState, mStateDot);

            // F_{,x}
            auto t_dFdx = mObjective->gradient_x(tSolution, aControl, mTimeStep);

            auto tLastStepIndex = mNumSteps - 1;
            for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--) {

                auto tU = Kokkos::subview(mState, tStepIndex, Kokkos::ALL());
                auto tV = Kokkos::subview(mStateDot, tStepIndex, Kokkos::ALL());

                Plato::ScalarVector tAdjoint_U = Kokkos::subview(mAdjoints_U, tStepIndex, Kokkos::ALL());
                Plato::ScalarVector tAdjoint_V = Kokkos::subview(mAdjoints_V, tStepIndex, Kokkos::ALL());

                // F_{,u^k}
                auto t_dFdu = mObjective->gradient_u(tSolution, aControl, tStepIndex, mTimeStep);
                // F_{,v^k}
                auto t_dFdv = mObjective->gradient_v(tSolution, aControl, tStepIndex, mTimeStep);

                if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1

                    // L_{v}^{k+1}
                    Plato::ScalarVector tAdjoint_V_next = Kokkos::subview(mAdjoints_V, tStepIndex+1, Kokkos::ALL());


                    // R_{v,u^k}^{k+1}
                    auto tR_vu_prev = mTrapezoidIntegrator.v_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{v}^{k+1} R_{v,u^k}^{k+1}
                    Plato::blas1::axpy(tR_vu_prev, tAdjoint_V_next, t_dFdu);


                    // R_{v,v^k}^{k+1}
                    auto tR_vv_prev = mTrapezoidIntegrator.v_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{v}^{k+1} R_{v,v^k}^{k+1}
                    Plato::blas1::axpy(tR_vv_prev, tAdjoint_V_next, t_dFdv);

                }
                Plato::blas1::scale(static_cast<Plato::Scalar>(-1), t_dFdu);

                // R_{v,u^k}
                auto tR_vu = mTrapezoidIntegrator.v_grad_u(mTimeStep);

                // -F_{,u^k} += R_{v,u^k}^k F_{,v^k}
                Plato::blas1::axpy(tR_vu, t_dFdv, t_dFdu);


                // R_{u,u^k}
                mJacobianU = mPDEConstraint.gradient_u(tU, tV, aControl, mTimeStep);

                // R_{u,v^k}
                mJacobianV = mPDEConstraint.gradient_v(tU, tV, aControl, mTimeStep);

                // R_{u,u^k} -= R_{v,u^k} R_{u,v^k}
                Plato::blas1::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

                this->applyConstraints(mJacobianU, t_dFdu);

                // L_u^k
                mSolver->solve(*mJacobianU, tAdjoint_U, t_dFdu);

                // L_v^k
                Plato::MatrixTimesVectorPlusVector(mJacobianV, tAdjoint_U, t_dFdv);
                Plato::blas1::fill(0.0, tAdjoint_V);
                Plato::blas1::axpy(-1.0, t_dFdv, tAdjoint_V);

                // R^k_{,x}
                auto t_dRdx = mPDEConstraint.gradient_x(tU, tV, aControl, mTimeStep);

                // F_{,x} += L_u^k R^k_{,x}
                Plato::MatrixTimesVectorPlusVector(t_dRdx, tAdjoint_U, t_dFdx);
            }

            return t_dFdx;
        }
    };

} // namespace Parabolic

} // namespace Plato
