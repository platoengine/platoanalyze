#ifndef PLATO_HYPERBOLIC_PROBLEM_HPP
#define PLATO_HYPERBOLIC_PROBLEM_HPP

#include "BLAS1.hpp"
#include "EssentialBCs.hpp"
#include "AnalyzeMacros.hpp"
#include "SimplexMechanics.hpp"
#include "PlatoAbstractProblem.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "ComputedField.hpp"

#include "hyperbolic/Newmark.hpp"
#include "hyperbolic/HyperbolicMechanics.hpp"
#include "hyperbolic/HyperbolicVectorFunction.hpp"
#include "hyperbolic/HyperbolicScalarFunctionBase.hpp"
#include "hyperbolic/HyperbolicScalarFunctionFactory.hpp"

namespace Plato
{

    template<typename SimplexPhysics>
    class HyperbolicProblem: public Plato::AbstractProblem
    {
      private:
        static constexpr Plato::OrdinalType SpatialDim = SimplexPhysics::mNumSpatialDims;
        static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;


        Plato::Hyperbolic::VectorFunction<SimplexPhysics> mPDEConstraint;

        Plato::NewmarkIntegrator<SimplexPhysics>     mNewmarkIntegrator;

        Plato::OrdinalType mNumSteps;
        Plato::Scalar      mTimeStep;

        bool mSaveState;

        std::shared_ptr<const Plato::Hyperbolic::ScalarFunctionBase> mObjective;

        Plato::ScalarVector      mResidual;
        Plato::ScalarVector      mResidualV;
        Plato::ScalarVector      mResidualA;

        Plato::ScalarMultiVector mAdjoints_U;
        Plato::ScalarMultiVector mAdjoints_V;
        Plato::ScalarMultiVector mAdjoints_A;

        Plato::ScalarMultiVector mDisplacement;
        Plato::ScalarMultiVector mVelocity;
        Plato::ScalarMultiVector mAcceleration;

        Teuchos::RCP<Plato::CrsMatrixType> mJacobianU;
        Teuchos::RCP<Plato::CrsMatrixType> mJacobianV;
        Teuchos::RCP<Plato::CrsMatrixType> mJacobianA;

        Teuchos::RCP<Plato::ComputedFields<SpatialDim>> mComputedFields;

        Plato::EssentialBCs<SimplexPhysics> mStateBoundaryConditions;

        Plato::LocalOrdinalVector mStateBcDofs;
        Plato::ScalarVector mStateBcValues;

        rcp<Plato::AbstractSolver> mSolver;
      public:
        /******************************************************************************/
        HyperbolicProblem(
          Omega_h::Mesh& aMesh,
          Omega_h::MeshSets& aMeshSets,
          Teuchos::ParameterList& aParamList,
          Comm::Machine aMachine
        ) :
            mPDEConstraint   (aMesh, aMeshSets, mDataMap, aParamList,
                                   aParamList.get<std::string>("PDE Constraint")),
            mNewmarkIntegrator    (aParamList.sublist("Time Integration")),
            mNumSteps     (aParamList.sublist("Time Integration").get<int>("Number Time Steps")),
            mTimeStep     (aParamList.sublist("Time Integration").get<Plato::Scalar>("Time Step")),
            mSaveState    (aParamList.sublist("Hyperbolic").isType<Teuchos::Array<std::string>>("Plottable")),
            mObjective    (nullptr),
            mResidual     ("MyResidual", mPDEConstraint.size()),
            mDisplacement ("Displacement", mNumSteps, mPDEConstraint.size()),
            mVelocity     ("Velocity",     mNumSteps, mPDEConstraint.size()),
            mAcceleration ("Acceleration", mNumSteps, mPDEConstraint.size()),
            mJacobianU(Teuchos::null),
            mJacobianV(Teuchos::null),
            mJacobianA(Teuchos::null),
            mStateBoundaryConditions(aParamList.sublist("Displacement Boundary Conditions",false), aMeshSets)
        /******************************************************************************/
        {

            // parse objective
            //
            if(aParamList.isType<std::string>("Objective"))
            {
                std::string tName = aParamList.get<std::string>("Objective");
                Plato::Hyperbolic::ScalarFunctionFactory<SimplexPhysics> tScalarFunctionFactory;
                mObjective = tScalarFunctionFactory.create(aMesh, aMeshSets, mDataMap, aParamList, tName);

                auto tLength = mPDEConstraint.size();
                mAdjoints_U = Plato::ScalarMultiVector("MyAdjoint U", mNumSteps, tLength);
                mAdjoints_V = Plato::ScalarMultiVector("MyAdjoint V", mNumSteps, tLength);
                mAdjoints_A = Plato::ScalarMultiVector("MyAdjoint A", mNumSteps, tLength);
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
                if(mComputedFields == Teuchos::null) {
                  THROWERR("No 'Computed Fields' have been defined");
                }

                Plato::ScalarVector tInitialState = Kokkos::subview(mDisplacement, 0, Kokkos::ALL());
                Plato::ScalarVector tInitialStateDot = Kokkos::subview(mVelocity, 0, Kokkos::ALL());

                auto tDofNames = mPDEConstraint.getDofNames();
                auto tDofDotNames = mPDEConstraint.getDofDotNames();

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
                               break;
                            }
                        }
                        if (tDofIndex != -1)
                        {
                            mComputedFields->get(tFieldName, tDofIndex, tDofNames.size(), tInitialState);
                        }
                        else
                        {
                            for (int j = 0; j < tDofDotNames.size(); ++j)
                            {
                                if (tDofDotNames[j] == tName) {
                                   tDofIndex = j;
                                   break;
                                }
                            }
                            if (tDofIndex != -1)
                            {
                                mComputedFields->get(tFieldName, tDofIndex, tDofDotNames.size(), tInitialStateDot);
                            }
                            else
                            {
                                THROWERR("Attempted to initialize state variable that doesn't exist.");
                            }
                        }
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
            return Plato::Solution(mDisplacement, mVelocity, mAcceleration);
        }

        /******************************************************************************/
        Plato::Adjoint getAdjoint()
        /******************************************************************************/
        {
            return Plato::Adjoint(mAdjoints_U, mAdjoints_V, mAdjoints_A);
        }
        /******************************************************************************/
        void setGlobalSolution(const Plato::Solution & aSolution)
        /******************************************************************************/
        {
            auto tState = aSolution.State;
            assert(tState.extent(0) == mDisplacement.extent(0));
            assert(tState.extent(1) == mDisplacement.extent(1));
            Kokkos::deep_copy(mDisplacement, tState);

            auto tStateDot = aSolution.StateDot;
            assert(tStateDot.extent(0) == mVelocity.extent(0));
            assert(tStateDot.extent(1) == mVelocity.extent(1));
            Kokkos::deep_copy(mVelocity, tStateDot);

            auto tStateDotDot = aSolution.StateDotDot;
            assert(tStateDotDot.extent(0) == mAcceleration.extent(0));
            assert(tStateDotDot.extent(1) == mAcceleration.extent(1));
            Kokkos::deep_copy(mAcceleration, tStateDotDot);
        }

        /******************************************************************************/
        void applyConstraints(
          const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
          const Plato::ScalarVector & aVector
        )
        /******************************************************************************/
        {
            if(mJacobianU->isBlockMatrix())
            {
                Plato::applyBlockConstraints<mNumDofsPerNode>(aMatrix, aVector, mStateBcDofs, mStateBcValues);
            }
            else
            {
                Plato::applyConstraints<mNumDofsPerNode>(aMatrix, aVector, mStateBcDofs, mStateBcValues);
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
        solution(const Plato::ScalarVector & aControl)
        /******************************************************************************/
        {
            mDataMap.clearStates();
            Plato::ScalarVector tDisplacementInit = Kokkos::subview(mDisplacement, /*StepIndex=*/0, Kokkos::ALL());
            Plato::ScalarVector tVelocityInit     = Kokkos::subview(mVelocity,     /*StepIndex=*/0, Kokkos::ALL());
            Plato::ScalarVector tAccelerationInit = Kokkos::subview(mAcceleration, /*StepIndex=*/0, Kokkos::ALL());
            mResidual  = mPDEConstraint.value(tDisplacementInit, tVelocityInit, tAccelerationInit, aControl, mTimeStep, 0.0);
            mDataMap.saveState();

            Plato::Scalar tCurrentTime(0.0);
            for(Plato::OrdinalType tStepIndex = 1; tStepIndex < mNumSteps; tStepIndex++) {
              tCurrentTime += mTimeStep;
              Plato::ScalarVector tDisplacementPrev = Kokkos::subview(mDisplacement, tStepIndex-1, Kokkos::ALL());
              Plato::ScalarVector tVelocityPrev     = Kokkos::subview(mVelocity,     tStepIndex-1, Kokkos::ALL());
              Plato::ScalarVector tAccelerationPrev = Kokkos::subview(mAcceleration, tStepIndex-1, Kokkos::ALL());

              Plato::ScalarVector tDisplacement = Kokkos::subview(mDisplacement, tStepIndex, Kokkos::ALL());
              Plato::ScalarVector tVelocity     = Kokkos::subview(mVelocity,     tStepIndex, Kokkos::ALL());
              Plato::ScalarVector tAcceleration = Kokkos::subview(mAcceleration, tStepIndex, Kokkos::ALL());


              // -R_{u}
              mResidual  = mPDEConstraint.value(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, tCurrentTime);
              Plato::blas1::scale(-1.0, mResidual);

              // R_{v}
              mResidualV = mNewmarkIntegrator.v_value(tDisplacement, tDisplacementPrev,
                                                      tVelocity,     tVelocityPrev,
                                                      tAcceleration, tAccelerationPrev, mTimeStep);

              // R_{u,v^N}
              mJacobianV = mPDEConstraint.gradient_v(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, tCurrentTime);

              // -R_{u} += R_{u,v^N} R_{v}
              Plato::MatrixTimesVectorPlusVector(mJacobianV, mResidualV, mResidual);

              // R_{a}
              mResidualA = mNewmarkIntegrator.a_value(tDisplacement, tDisplacementPrev,
                                                      tVelocity,     tVelocityPrev,
                                                      tAcceleration, tAccelerationPrev, mTimeStep);

              // R_{u,a^N}
              mJacobianA = mPDEConstraint.gradient_a(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, tCurrentTime);

              // -R_{u} += R_{u,a^N} R_{a}
              Plato::MatrixTimesVectorPlusVector(mJacobianA, mResidualA, mResidual);

              // R_{u,u^N}
              mJacobianU = mPDEConstraint.gradient_u(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, tCurrentTime);

              // R_{v,u^N}
              auto tR_vu = mNewmarkIntegrator.v_grad_u(mTimeStep);

              // R_{u,u^N} += R_{u,v^N} R_{v,u^N}
              Plato::blas1::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

              // R_{a,u^N}
              auto tR_au = mNewmarkIntegrator.a_grad_u(mTimeStep);

              // R_{u,u^N} += R_{u,a^N} R_{a,u^N}
              Plato::blas1::axpy(-tR_au, mJacobianA->entries(), mJacobianU->entries());

              mStateBoundaryConditions.get(mStateBcDofs, mStateBcValues, tCurrentTime);
              this->applyConstraints(mJacobianU, mResidual);

              Plato::ScalarVector tDeltaD("increment", tDisplacement.extent(0));
              Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaD);

              // compute displacement increment:
              mSolver->solve(*mJacobianU, tDeltaD, mResidual);

              // compute and add velocity increment: \Delta v = - ( R_{v} + R_{v,u} \Delta u )
              Plato::blas1::axpy(tR_vu, tDeltaD, mResidualV);
              // a_{k+1} = a_{k} + \Delta a
              Plato::blas1::axpy(-1.0, mResidualV, tVelocity);

              // compute and add acceleration increment: \Delta a = - ( R_{a} + R_{a,u} \Delta u )
              Plato::blas1::axpy(tR_au, tDeltaD, mResidualA);
              // a_{k+1} = a_{k} + \Delta a
              Plato::blas1::axpy(-1.0, mResidualA, tAcceleration);

              // add displacement increment
              Plato::blas1::axpy(1.0, tDeltaD, tDisplacement);

              if ( mSaveState )
              {
                // evaluate at new state
                mResidual  = mPDEConstraint.value(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, tCurrentTime);
                mDataMap.saveState();
              }
            }
            return Plato::Solution(mDisplacement, mVelocity, mAcceleration);
        }

        /******************************************************************************/
        Plato::Scalar constraintValue(
          const Plato::ScalarVector & aControl,
          const Plato::Solution & aSolution
        )
        /******************************************************************************/
        {
            return 0;
        }
        /******************************************************************************/
        Plato::Scalar constraintValue(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            return 0;
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradient(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("constraint gradient", 0);
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradient(
          const Plato::ScalarVector & aControl,
          const Plato::Solution & aSolution
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("constraint gradient", 0);
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradientX(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("constraint gradientX", 0);
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradientX(
          const Plato::ScalarVector & aControl,
          const Plato::Solution & aSolution
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("constraint gradientX", 0);
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
                THROWERR("No objective is defined in the input file.");
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
                THROWERR("No objective is defined in the input file.");
            }
            auto tSolution = Plato::Solution(mDisplacement, mVelocity, mAcceleration);
            return mObjective->value(tSolution, aControl, mTimeStep);
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradient(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            if(mObjective == nullptr)
            {
                THROWERR("No objective is defined in the input file.");
            }
            auto tSolution = solution(aControl);
            return objectiveGradient(aControl, tSolution);
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradient(
          const Plato::ScalarVector & aControl,
          const Plato::Solution     & aSolution
        )
        /******************************************************************************/
        {
            if(mObjective == nullptr)
            {
                THROWERR("No objective is defined in the input file.");
            }

            auto tSolution = Plato::Solution(mDisplacement, mVelocity, mAcceleration);

            // F_{,z}
            auto t_dFdz = mObjective->gradient_z(tSolution, aControl, mTimeStep);

            auto tLastStepIndex = mNumSteps - 1;
            Plato::Scalar tCurrentTime(mTimeStep*mNumSteps);
            for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--)
            {
                auto tU = Kokkos::subview(mDisplacement, tStepIndex, Kokkos::ALL());
                auto tV = Kokkos::subview(mVelocity,     tStepIndex, Kokkos::ALL());
                auto tA = Kokkos::subview(mAcceleration, tStepIndex, Kokkos::ALL());

                Plato::ScalarVector tAdjoint_U = Kokkos::subview(mAdjoints_U, tStepIndex, Kokkos::ALL());
                Plato::ScalarVector tAdjoint_V = Kokkos::subview(mAdjoints_V, tStepIndex, Kokkos::ALL());
                Plato::ScalarVector tAdjoint_A = Kokkos::subview(mAdjoints_A, tStepIndex, Kokkos::ALL());

                // F_{,u^k}
                auto t_dFdu = mObjective->gradient_u(tSolution, aControl, tStepIndex, mTimeStep);
                // F_{,v^k}
                auto t_dFdv = mObjective->gradient_v(tSolution, aControl, tStepIndex, mTimeStep);
                // F_{,a^k}
                auto t_dFda = mObjective->gradient_a(tSolution, aControl, tStepIndex, mTimeStep);

                if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1

                    // L_{v}^{k+1}
                    Plato::ScalarVector tAdjoint_V_next = Kokkos::subview(mAdjoints_V, tStepIndex+1, Kokkos::ALL());

                    // L_{a}^{k+1}
                    Plato::ScalarVector tAdjoint_A_next = Kokkos::subview(mAdjoints_A, tStepIndex+1, Kokkos::ALL());


                    // R_{v,u^k}^{k+1}
                    auto tR_vu_prev = mNewmarkIntegrator.v_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{v}^{k+1} R_{v,u^k}^{k+1}
                    Plato::blas1::axpy(tR_vu_prev, tAdjoint_V_next, t_dFdu);

                    // R_{a,u^k}^{k+1}
                    auto tR_au_prev = mNewmarkIntegrator.a_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{a}^{k+1} R_{a,u^k}^{k+1}
                    Plato::blas1::axpy(tR_au_prev, tAdjoint_A_next, t_dFdu);


                    // R_{v,v^k}^{k+1}
                    auto tR_vv_prev = mNewmarkIntegrator.v_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{v}^{k+1} R_{v,v^k}^{k+1}
                    Plato::blas1::axpy(tR_vv_prev, tAdjoint_V_next, t_dFdv);

                    // R_{a,v^k}^{k+1}
                    auto tR_av_prev = mNewmarkIntegrator.a_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{a}^{k+1} R_{a,v^k}^{k+1}
                    Plato::blas1::axpy(tR_av_prev, tAdjoint_A_next, t_dFdv);


                    // R_{v,a^k}^{k+1}
                    auto tR_va_prev = mNewmarkIntegrator.v_grad_a_prev(mTimeStep);

                    // F_{,a^k} += L_{v}^{k+1} R_{v,a^k}^{k+1}
                    Plato::blas1::axpy(tR_va_prev, tAdjoint_V_next, t_dFda);

                    // R_{a,a^k}^{k+1}
                    auto tR_aa_prev = mNewmarkIntegrator.a_grad_a_prev(mTimeStep);

                    // F_{,a^k} += L_{a}^{k+1} R_{a,a^k}^{k+1}
                    Plato::blas1::axpy(tR_aa_prev, tAdjoint_A_next, t_dFda);

                }
                Plato::blas1::scale(static_cast<Plato::Scalar>(-1), t_dFdu);

                // R_{v,u^k}
                auto tR_vu = mNewmarkIntegrator.v_grad_u(mTimeStep);

                // -F_{,u^k} += R_{v,u^k}^k F_{,v^k}
                Plato::blas1::axpy(tR_vu, t_dFdv, t_dFdu);

                // R_{a,u^k}
                auto tR_au = mNewmarkIntegrator.a_grad_u(mTimeStep);

                // -F_{,u^k} += R_{a,u^k}^k F_{,a^k}
                Plato::blas1::axpy(tR_au, t_dFda, t_dFdu);

                // R_{u,u^k}
                mJacobianU = mPDEConstraint.gradient_u(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

                // R_{u,v^k}
                mJacobianV = mPDEConstraint.gradient_v(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

                // R_{u,a^k}
                mJacobianA = mPDEConstraint.gradient_a(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

                // R_{u,u^k} -= R_{v,u^k} R_{u,v^k}
                Plato::blas1::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

                // R_{u,u^k} -= R_{a,u^k} R_{u,a^k}
                Plato::blas1::axpy(-tR_au, mJacobianA->entries(), mJacobianU->entries());

                this->applyConstraints(mJacobianU, t_dFdu);

                // L_u^k
                mSolver->solve(*mJacobianU, tAdjoint_U, t_dFdu);

                // L_v^k
                Plato::MatrixTimesVectorPlusVector(mJacobianV, tAdjoint_U, t_dFdv);
                Plato::blas1::fill(0.0, tAdjoint_V);
                Plato::blas1::axpy(-1.0, t_dFdv, tAdjoint_V);

                // L_a^k
                Plato::MatrixTimesVectorPlusVector(mJacobianA, tAdjoint_U, t_dFda);
                Plato::blas1::fill(0.0, tAdjoint_A);
                Plato::blas1::axpy(-1.0, t_dFda, tAdjoint_A);

                // R^k_{,z}
                auto t_dRdz = mPDEConstraint.gradient_z(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

                // F_{,z} += L_u^k R^k_{,z}
                Plato::MatrixTimesVectorPlusVector(t_dRdz, tAdjoint_U, t_dFdz);

                tCurrentTime -= mTimeStep;
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
                THROWERR("No objective is defined in the input file.");
            }
            auto tSolution = solution(aControl);
            return objectiveGradientX(aControl, tSolution);
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradientX(
          const Plato::ScalarVector & aControl,
          const Plato::Solution     & aSolution
        )
        /******************************************************************************/
        {
            if(mObjective == nullptr)
            {
                THROWERR("No objective is defined in the input file.");
            }

            auto tSolution = Plato::Solution(mDisplacement, mVelocity, mAcceleration);

            // F_{,x}
            auto t_dFdx = mObjective->gradient_x(tSolution, aControl, mTimeStep);

            auto tLastStepIndex = mNumSteps - 1;
            Plato::Scalar tCurrentTime(mTimeStep*mNumSteps);
            for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--) {

                auto tU = Kokkos::subview(mDisplacement, tStepIndex, Kokkos::ALL());
                auto tV = Kokkos::subview(mVelocity,     tStepIndex, Kokkos::ALL());
                auto tA = Kokkos::subview(mAcceleration, tStepIndex, Kokkos::ALL());

                Plato::ScalarVector tAdjoint_U = Kokkos::subview(mAdjoints_U, tStepIndex, Kokkos::ALL());
                Plato::ScalarVector tAdjoint_V = Kokkos::subview(mAdjoints_V, tStepIndex, Kokkos::ALL());
                Plato::ScalarVector tAdjoint_A = Kokkos::subview(mAdjoints_A, tStepIndex, Kokkos::ALL());

                // F_{,u^k}
                auto t_dFdu = mObjective->gradient_u(tSolution, aControl, tStepIndex, mTimeStep);
                // F_{,v^k}
                auto t_dFdv = mObjective->gradient_v(tSolution, aControl, tStepIndex, mTimeStep);
                // F_{,a^k}
                auto t_dFda = mObjective->gradient_a(tSolution, aControl, tStepIndex, mTimeStep);

                if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1

                    // L_{v}^{k+1}
                    Plato::ScalarVector tAdjoint_V_next = Kokkos::subview(mAdjoints_V, tStepIndex+1, Kokkos::ALL());

                    // L_{a}^{k+1}
                    Plato::ScalarVector tAdjoint_A_next = Kokkos::subview(mAdjoints_A, tStepIndex+1, Kokkos::ALL());


                    // R_{v,u^k}^{k+1}
                    auto tR_vu_prev = mNewmarkIntegrator.v_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{v}^{k+1} R_{v,u^k}^{k+1}
                    Plato::blas1::axpy(tR_vu_prev, tAdjoint_V_next, t_dFdu);

                    // R_{a,u^k}^{k+1}
                    auto tR_au_prev = mNewmarkIntegrator.a_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{a}^{k+1} R_{a,u^k}^{k+1}
                    Plato::blas1::axpy(tR_au_prev, tAdjoint_A_next, t_dFdu);


                    // R_{v,v^k}^{k+1}
                    auto tR_vv_prev = mNewmarkIntegrator.v_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{v}^{k+1} R_{v,v^k}^{k+1}
                    Plato::blas1::axpy(tR_vv_prev, tAdjoint_V_next, t_dFdv);

                    // R_{a,v^k}^{k+1}
                    auto tR_av_prev = mNewmarkIntegrator.a_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{a}^{k+1} R_{a,v^k}^{k+1}
                    Plato::blas1::axpy(tR_av_prev, tAdjoint_A_next, t_dFdv);


                    // R_{v,a^k}^{k+1}
                    auto tR_va_prev = mNewmarkIntegrator.v_grad_a_prev(mTimeStep);

                    // F_{,a^k} += L_{v}^{k+1} R_{v,a^k}^{k+1}
                    Plato::blas1::axpy(tR_va_prev, tAdjoint_V_next, t_dFda);

                    // R_{a,a^k}^{k+1}
                    auto tR_aa_prev = mNewmarkIntegrator.a_grad_a_prev(mTimeStep);

                    // F_{,a^k} += L_{a}^{k+1} R_{a,a^k}^{k+1}
                    Plato::blas1::axpy(tR_aa_prev, tAdjoint_A_next, t_dFda);

                }
                Plato::blas1::scale(static_cast<Plato::Scalar>(-1), t_dFdu);

                // R_{v,u^k}
                auto tR_vu = mNewmarkIntegrator.v_grad_u(mTimeStep);

                // -F_{,u^k} += R_{v,u^k}^k F_{,v^k}
                Plato::blas1::axpy(tR_vu, t_dFdv, t_dFdu);

                // R_{a,u^k}
                auto tR_au = mNewmarkIntegrator.a_grad_u(mTimeStep);

                // -F_{,u^k} += R_{a,u^k}^k F_{,a^k}
                Plato::blas1::axpy(tR_au, t_dFda, t_dFdu);

                // R_{u,u^k}
                mJacobianU = mPDEConstraint.gradient_u(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

                // R_{u,v^k}
                mJacobianV = mPDEConstraint.gradient_v(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

                // R_{u,a^k}
                mJacobianA = mPDEConstraint.gradient_a(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

                // R_{u,u^k} -= R_{v,u^k} R_{u,v^k}
                Plato::blas1::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

                // R_{u,u^k} -= R_{a,u^k} R_{u,a^k}
                Plato::blas1::axpy(-tR_au, mJacobianA->entries(), mJacobianU->entries());

                this->applyConstraints(mJacobianU, t_dFdu);

                // L_u^k
                mSolver->solve(*mJacobianU, tAdjoint_U, t_dFdu);

                // L_v^k
                Plato::MatrixTimesVectorPlusVector(mJacobianV, tAdjoint_U, t_dFdv);
                Plato::blas1::fill(0.0, tAdjoint_V);
                Plato::blas1::axpy(-1.0, t_dFdv, tAdjoint_V);

                // L_a^k
                Plato::MatrixTimesVectorPlusVector(mJacobianA, tAdjoint_U, t_dFda);
                Plato::blas1::fill(0.0, tAdjoint_A);
                Plato::blas1::axpy(-1.0, t_dFda, tAdjoint_A);

                // R^k_{,x}
                auto t_dRdx = mPDEConstraint.gradient_x(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

                // F_{,x} += L_u^k R^k_{,x}
                Plato::MatrixTimesVectorPlusVector(t_dRdx, tAdjoint_U, t_dFdx);

                tCurrentTime -= mTimeStep;
            }

            return t_dFdx;
        }
    };
}

#endif
