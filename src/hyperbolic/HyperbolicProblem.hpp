#ifndef PLATO_HYPERBOLIC_PROBLEM_HPP
#define PLATO_HYPERBOLIC_PROBLEM_HPP

#include "BLAS1.hpp"
#include "Solutions.hpp"
#include "EssentialBCs.hpp"
#include "SpatialModel.hpp"
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

        using Criterion = std::shared_ptr<Plato::Hyperbolic::ScalarFunctionBase>;
        using Criteria  = std::map<std::string, Criterion>;

        static constexpr Plato::OrdinalType SpatialDim = SimplexPhysics::mNumSpatialDims;
        static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;

        Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

        using VectorFunctionType = Plato::Hyperbolic::VectorFunction<SimplexPhysics>;

        VectorFunctionType mPDEConstraint;

        Plato::NewmarkIntegrator<SimplexPhysics> mIntegrator;

        Plato::OrdinalType mNumSteps;
        Plato::Scalar      mTimeStep;

        bool mSaveState;

        Criteria mCriteria;

        Plato::ScalarVector      mResidual;
        Plato::ScalarVector      mResidualV;
        Plato::ScalarVector      mResidualA;

        Plato::ScalarMultiVector mAdjoints_U;
        Plato::ScalarMultiVector mAdjoints_V;
        Plato::ScalarMultiVector mAdjoints_A;

        Plato::ScalarMultiVector mDisplacement;
        Plato::ScalarMultiVector mVelocity;
        Plato::ScalarMultiVector mAcceleration;

        Plato::ScalarVector mInitDisplacement;
        Plato::ScalarVector mInitVelocity;
        Plato::ScalarVector mInitAcceleration;

        Teuchos::RCP<Plato::CrsMatrixType> mJacobianU;
        Teuchos::RCP<Plato::CrsMatrixType> mJacobianV;
        Teuchos::RCP<Plato::CrsMatrixType> mJacobianA;

        Teuchos::RCP<Plato::ComputedFields<SpatialDim>> mComputedFields;

        Plato::EssentialBCs<SimplexPhysics> mStateBoundaryConditions;

        Plato::LocalOrdinalVector mStateBcDofs;
        Plato::ScalarVector mStateBcValues;

        rcp<Plato::AbstractSolver> mSolver;
        std::string mPhysics; /*!< physics used for the simulation */

      public:
        /******************************************************************************/
        HyperbolicProblem(
          Omega_h::Mesh& aMesh,
          Omega_h::MeshSets& aMeshSets,
          Teuchos::ParameterList& aProblemParams,
          Comm::Machine aMachine
        ) :
            mSpatialModel  (aMesh, aMeshSets, aProblemParams),
            mPDEConstraint (mSpatialModel, mDataMap, aProblemParams, aProblemParams.get<std::string>("PDE Constraint")),
            mIntegrator    (aProblemParams.sublist("Time Integration")),
            mNumSteps      (aProblemParams.sublist("Time Integration").get<int>("Number Time Steps")),
            mTimeStep      (aProblemParams.sublist("Time Integration").get<Plato::Scalar>("Time Step")),
            mSaveState     (aProblemParams.sublist("Hyperbolic").isType<Teuchos::Array<std::string>>("Plottable")),
            mResidual      ("MyResidual", mPDEConstraint.size()),
            mDisplacement  ("Displacement", mNumSteps, mPDEConstraint.size()),
            mVelocity      ("Velocity",     mNumSteps, mPDEConstraint.size()),
            mAcceleration  ("Acceleration", mNumSteps, mPDEConstraint.size()),
            mInitDisplacement ("Init Displacement", mPDEConstraint.size()),
            mInitVelocity     ("Init Velocity",     mPDEConstraint.size()),
            mInitAcceleration ("Init Acceleration", mPDEConstraint.size()),
            mJacobianU     (Teuchos::null),
            mJacobianV     (Teuchos::null),
            mJacobianA     (Teuchos::null),
            mStateBoundaryConditions(aProblemParams.sublist("Displacement Boundary Conditions",false), aMeshSets),
            mPhysics       (aProblemParams.get<std::string>("Physics"))
        /******************************************************************************/
        {
            // parse criteria
            //
            if(aProblemParams.isSublist("Criteria"))
            {
                Plato::Hyperbolic::ScalarFunctionFactory<SimplexPhysics> tFunctionBaseFactory;

                auto tCriteriaParams = aProblemParams.sublist("Criteria");
                for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaParams.begin(); tIndex != tCriteriaParams.end(); ++tIndex)
                {
                    const Teuchos::ParameterEntry & tEntry = tCriteriaParams.entry(tIndex);
                    std::string tName = tCriteriaParams.name(tIndex);

                    TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error,
                      " Parameter in Criteria block not valid.  Expect lists only.");

                    {
                        auto tCriterion = tFunctionBaseFactory.create(mSpatialModel, mDataMap, aProblemParams, tName);
                        if( tCriterion != nullptr )
                        {
                            mCriteria[tName] = tCriterion;
                        }
                    }
                }
                if( mCriteria.size() )
                {
                    auto tLength = mPDEConstraint.size();
                    mAdjoints_U = Plato::ScalarMultiVector("MyAdjoint U", mNumSteps, tLength);
                    mAdjoints_V = Plato::ScalarMultiVector("MyAdjoint V", mNumSteps, tLength);
                    mAdjoints_A = Plato::ScalarMultiVector("MyAdjoint A", mNumSteps, tLength);
                }
            }

            // parse computed fields
            //
            if(aProblemParams.isSublist("Computed Fields"))
            {
              mComputedFields = Teuchos::rcp(new Plato::ComputedFields<SpatialDim>(aMesh, aProblemParams.sublist("Computed Fields")));
            }

            // parse initial state
            //
            if(aProblemParams.isSublist("Initial State"))
            {
                if(mComputedFields == Teuchos::null) {
                  THROWERR("No 'Computed Fields' have been defined");
                }

                auto tDofNames = mPDEConstraint.getDofNames();
                auto tDofDotNames = mPDEConstraint.getDofDotNames();

                auto tInitStateParams = aProblemParams.sublist("Initial State");
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
                            mComputedFields->get(tFieldName, tDofIndex, tDofNames.size(), mInitDisplacement);
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
                                mComputedFields->get(tFieldName, tDofIndex, tDofDotNames.size(), mInitVelocity);
                            }
                            else
                            {
                                THROWERR("Attempted to initialize state variable that doesn't exist.");
                            }
                        }
                    }
                }
            }

            Plato::SolverFactory tSolverFactory(aProblemParams.sublist("Linear Solver"));

            mSolver = tSolverFactory.create(aMesh, aMachine, SimplexPhysics::mNumDofsPerNode);

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

        /******************************************************************************//**
         * \brief Update physics-based parameters within optimization iterations
         * \param [in] aState 2D container of state variables
         * \param [in] aControl 1D container of control variables
        **********************************************************************************/
        void updateProblem(const Plato::ScalarVector & aControl, const Plato::Solutions & aSolution)
        { return; }
        /******************************************************************************/
        Plato::Solutions
        solution(const Plato::ScalarVector & aControl)
        /******************************************************************************/
        {
            mDataMap.clearStates();
            Kokkos::deep_copy(mDisplacement, 0.0);
            Kokkos::deep_copy(mVelocity,     0.0);
            Kokkos::deep_copy(mAcceleration, 0.0);
            Plato::ScalarVector tDisplacementInit = Kokkos::subview(mDisplacement, /*StepIndex=*/0, Kokkos::ALL());
            Plato::ScalarVector tVelocityInit     = Kokkos::subview(mVelocity,     /*StepIndex=*/0, Kokkos::ALL());
            Plato::ScalarVector tAccelerationInit = Kokkos::subview(mAcceleration, /*StepIndex=*/0, Kokkos::ALL());
            Kokkos::deep_copy(tDisplacementInit, mInitDisplacement);
            Kokkos::deep_copy(tVelocityInit,     mInitVelocity);
            Kokkos::deep_copy(tAccelerationInit, mInitAcceleration);
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
              mResidualV = mIntegrator.v_value(tDisplacement, tDisplacementPrev,
                                                      tVelocity,     tVelocityPrev,
                                                      tAcceleration, tAccelerationPrev, mTimeStep);

              // R_{u,v^N}
              mJacobianV = mPDEConstraint.gradient_v(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, tCurrentTime);

              // -R_{u} += R_{u,v^N} R_{v}
              Plato::MatrixTimesVectorPlusVector(mJacobianV, mResidualV, mResidual);

              // R_{a}
              mResidualA = mIntegrator.a_value(tDisplacement, tDisplacementPrev,
                                                      tVelocity,     tVelocityPrev,
                                                      tAcceleration, tAccelerationPrev, mTimeStep);

              // R_{u,a^N}
              mJacobianA = mPDEConstraint.gradient_a(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, tCurrentTime);

              // -R_{u} += R_{u,a^N} R_{a}
              Plato::MatrixTimesVectorPlusVector(mJacobianA, mResidualA, mResidual);

              // R_{u,u^N}
              mJacobianU = mPDEConstraint.gradient_u(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, tCurrentTime);

              // R_{v,u^N}
              auto tR_vu = mIntegrator.v_grad_u(mTimeStep);

              // R_{u,u^N} += R_{u,v^N} R_{v,u^N}
              Plato::blas1::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

              // R_{a,u^N}
              auto tR_au = mIntegrator.a_grad_u(mTimeStep);

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

            auto tSolution = this->getSolution();
            return tSolution;
        }

        /******************************************************************************//**
         * \brief Evaluate criterion function
         * \param [in] aControl 1D view of control variables
         * \param [in] aSolution solution database
         * \param [in] aName Name of criterion.
         * \return criterion function value
        **********************************************************************************/
        Plato::Scalar
        criterionValue(
            const Plato::ScalarVector & aControl,
            const Plato::Solutions    & aSolution,
            const std::string         & aName
        ) override
        {
            if( mCriteria.count(aName) )
            {
                Criterion tCriterion = mCriteria[aName];
                return tCriterion->value(aSolution, aControl, mTimeStep);
            }
            else
            {
                THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }
        }

        /******************************************************************************//**
         * \brief Evaluate criterion function
         * \param [in] aControl 1D view of control variables
         * \param [in] aSolution solution database
         * \param [in] aName Name of criterion.
         * \return criterion function value
        **********************************************************************************/
        Plato::Scalar
        criterionValue(
            const Plato::ScalarVector & aControl,
            const std::string         & aName
        ) override
        {
            if( mCriteria.count(aName) )
            {
                auto tSolution = this->getSolution();
                Criterion tCriterion = mCriteria[aName];
                return tCriterion->value(tSolution, aControl, mTimeStep);
            }
            else
            {
                THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }
        }

        /******************************************************************************//**
         * \brief Evaluate criterion gradient wrt control variables
         * \param [in] aControl 1D view of control variables
         * \param [in] aSolution solution database
         * \param [in] aName Name of criterion.
         * \return 1D view - criterion gradient wrt control variables
        **********************************************************************************/
        Plato::ScalarVector
        criterionGradient(
            const Plato::ScalarVector & aControl,
            const std::string         & aName
        ) override
        {
            if( mCriteria.count(aName) )
            {
                auto tSolution = this->getSolution();
                Criterion tCriterion = mCriteria[aName];
                return criterionGradient(aControl, tSolution, tCriterion);
            }
            else
            {
                THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }
        }

        /******************************************************************************//**
         * \brief Evaluate criterion gradient wrt control variables
         * \param [in] aControl 1D view of control variables
         * \param [in] aSolution solution database
         * \param [in] aName Name of criterion.
         * \return 1D view - criterion gradient wrt control variables
        **********************************************************************************/
        Plato::ScalarVector
        criterionGradient(
            const Plato::ScalarVector & aControl,
            const Plato::Solutions    & aSolution,
            const std::string         & aName
        ) override
        {
            if( mCriteria.count(aName) )
            {
                Criterion tCriterion = mCriteria[aName];
                return criterionGradient(aControl, aSolution, tCriterion);
            }
            else
            {
                THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }
        }

        /******************************************************************************//**
         * \brief Evaluate criterion gradient wrt control variables
         * \param [in] aControl 1D view of control variables
         * \param [in] aSolution solution database
         * \param [in] aCriterion criterion to be evaluated
         * \return 1D view - criterion gradient wrt control variables
        **********************************************************************************/
        Plato::ScalarVector
        criterionGradient(
          const Plato::ScalarVector & aControl,
          const Plato::Solutions    & aSolution,
                Criterion             aCriterion
        )
        {
            if(aCriterion == nullptr)
            {
                THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }

            Plato::Solutions tSolution(mPhysics);
            tSolution.set("State", mDisplacement);
            tSolution.set("StateDot", mVelocity);
            tSolution.set("StateDotDot", mAcceleration);

            // F_{,z}
            auto t_dFdz = aCriterion->gradient_z(tSolution, aControl, mTimeStep);

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
                auto t_dFdu = aCriterion->gradient_u(tSolution, aControl, tStepIndex, mTimeStep);
                // F_{,v^k}
                auto t_dFdv = aCriterion->gradient_v(tSolution, aControl, tStepIndex, mTimeStep);
                // F_{,a^k}
                auto t_dFda = aCriterion->gradient_a(tSolution, aControl, tStepIndex, mTimeStep);

                if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1

                    // L_{v}^{k+1}
                    Plato::ScalarVector tAdjoint_V_next = Kokkos::subview(mAdjoints_V, tStepIndex+1, Kokkos::ALL());

                    // L_{a}^{k+1}
                    Plato::ScalarVector tAdjoint_A_next = Kokkos::subview(mAdjoints_A, tStepIndex+1, Kokkos::ALL());


                    // R_{v,u^k}^{k+1}
                    auto tR_vu_prev = mIntegrator.v_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{v}^{k+1} R_{v,u^k}^{k+1}
                    Plato::blas1::axpy(tR_vu_prev, tAdjoint_V_next, t_dFdu);

                    // R_{a,u^k}^{k+1}
                    auto tR_au_prev = mIntegrator.a_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{a}^{k+1} R_{a,u^k}^{k+1}
                    Plato::blas1::axpy(tR_au_prev, tAdjoint_A_next, t_dFdu);


                    // R_{v,v^k}^{k+1}
                    auto tR_vv_prev = mIntegrator.v_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{v}^{k+1} R_{v,v^k}^{k+1}
                    Plato::blas1::axpy(tR_vv_prev, tAdjoint_V_next, t_dFdv);

                    // R_{a,v^k}^{k+1}
                    auto tR_av_prev = mIntegrator.a_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{a}^{k+1} R_{a,v^k}^{k+1}
                    Plato::blas1::axpy(tR_av_prev, tAdjoint_A_next, t_dFdv);


                    // R_{v,a^k}^{k+1}
                    auto tR_va_prev = mIntegrator.v_grad_a_prev(mTimeStep);

                    // F_{,a^k} += L_{v}^{k+1} R_{v,a^k}^{k+1}
                    Plato::blas1::axpy(tR_va_prev, tAdjoint_V_next, t_dFda);

                    // R_{a,a^k}^{k+1}
                    auto tR_aa_prev = mIntegrator.a_grad_a_prev(mTimeStep);

                    // F_{,a^k} += L_{a}^{k+1} R_{a,a^k}^{k+1}
                    Plato::blas1::axpy(tR_aa_prev, tAdjoint_A_next, t_dFda);

                }
                Plato::blas1::scale(static_cast<Plato::Scalar>(-1), t_dFdu);

                // R_{v,u^k}
                auto tR_vu = mIntegrator.v_grad_u(mTimeStep);

                // -F_{,u^k} += R_{v,u^k}^k F_{,v^k}
                Plato::blas1::axpy(tR_vu, t_dFdv, t_dFdu);

                // R_{a,u^k}
                auto tR_au = mIntegrator.a_grad_u(mTimeStep);

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

        /******************************************************************************//**
         * \brief Evaluate criterion partial derivative wrt configuration variables
         * \param [in] aControl 1D view of control variables
         * \param [in] aName Name of criterion.
         * \return 1D view - criterion partial derivative wrt configuration variables
        **********************************************************************************/
        Plato::ScalarVector
        criterionGradientX(
            const Plato::ScalarVector & aControl,
            const std::string         & aName
        ) override
        {
            if( mCriteria.count(aName) )
            {
                auto tCriterion = mCriteria[aName];
                auto tSolution = this->getSolution();
                return criterionGradientX(aControl, tSolution, tCriterion);
            }
            else
            {
                THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }
        }

        /******************************************************************************//**
         * \brief Evaluate criterion gradient wrt configuration variables
         * \param [in] aControl 1D view of control variables
         * \param [in] aSolution solution database
         * \param [in] aName Name of criterion.
         * \return 1D view - criterion gradient wrt control variables
        **********************************************************************************/
        Plato::ScalarVector
        criterionGradientX(
            const Plato::ScalarVector & aControl,
            const Plato::Solutions    & aSolution,
            const std::string         & aName
        ) override
        {
            if( mCriteria.count(aName) )
            {
                Criterion tCriterion = mCriteria[aName];
                return criterionGradientX(aControl, aSolution, tCriterion);
            }
            else
            {
                THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }
        }

        /******************************************************************************//**
         * \brief Evaluate criterion gradient wrt configuration variables
         * \param [in] aControl 1D view of control variables
         * \param [in] aSolution solution database
         * \param [in] aCriterion criterion to be evaluated
         * \return 1D view - criterion gradient wrt control variables
        **********************************************************************************/
        Plato::ScalarVector
        criterionGradientX(
            const Plato::ScalarVector & aControl,
            const Plato::Solutions    & aSolution,
                  Criterion             aCriterion
        )
        {
            if(aCriterion == nullptr)
            {
                THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }

            Plato::Solutions tSolution(mPhysics);
            tSolution.set("State", mDisplacement);
            tSolution.set("StateDot", mVelocity);
            tSolution.set("StateDotDot", mAcceleration);

            // F_{,x}
            auto t_dFdx = aCriterion->gradient_x(tSolution, aControl, mTimeStep);

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
                auto t_dFdu = aCriterion->gradient_u(tSolution, aControl, tStepIndex, mTimeStep);
                // F_{,v^k}
                auto t_dFdv = aCriterion->gradient_v(tSolution, aControl, tStepIndex, mTimeStep);
                // F_{,a^k}
                auto t_dFda = aCriterion->gradient_a(tSolution, aControl, tStepIndex, mTimeStep);

                if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1

                    // L_{v}^{k+1}
                    Plato::ScalarVector tAdjoint_V_next = Kokkos::subview(mAdjoints_V, tStepIndex+1, Kokkos::ALL());

                    // L_{a}^{k+1}
                    Plato::ScalarVector tAdjoint_A_next = Kokkos::subview(mAdjoints_A, tStepIndex+1, Kokkos::ALL());


                    // R_{v,u^k}^{k+1}
                    auto tR_vu_prev = mIntegrator.v_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{v}^{k+1} R_{v,u^k}^{k+1}
                    Plato::blas1::axpy(tR_vu_prev, tAdjoint_V_next, t_dFdu);

                    // R_{a,u^k}^{k+1}
                    auto tR_au_prev = mIntegrator.a_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{a}^{k+1} R_{a,u^k}^{k+1}
                    Plato::blas1::axpy(tR_au_prev, tAdjoint_A_next, t_dFdu);


                    // R_{v,v^k}^{k+1}
                    auto tR_vv_prev = mIntegrator.v_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{v}^{k+1} R_{v,v^k}^{k+1}
                    Plato::blas1::axpy(tR_vv_prev, tAdjoint_V_next, t_dFdv);

                    // R_{a,v^k}^{k+1}
                    auto tR_av_prev = mIntegrator.a_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{a}^{k+1} R_{a,v^k}^{k+1}
                    Plato::blas1::axpy(tR_av_prev, tAdjoint_A_next, t_dFdv);


                    // R_{v,a^k}^{k+1}
                    auto tR_va_prev = mIntegrator.v_grad_a_prev(mTimeStep);

                    // F_{,a^k} += L_{v}^{k+1} R_{v,a^k}^{k+1}
                    Plato::blas1::axpy(tR_va_prev, tAdjoint_V_next, t_dFda);

                    // R_{a,a^k}^{k+1}
                    auto tR_aa_prev = mIntegrator.a_grad_a_prev(mTimeStep);

                    // F_{,a^k} += L_{a}^{k+1} R_{a,a^k}^{k+1}
                    Plato::blas1::axpy(tR_aa_prev, tAdjoint_A_next, t_dFda);

                }
                Plato::blas1::scale(static_cast<Plato::Scalar>(-1), t_dFdu);

                // R_{v,u^k}
                auto tR_vu = mIntegrator.v_grad_u(mTimeStep);

                // -F_{,u^k} += R_{v,u^k}^k F_{,v^k}
                Plato::blas1::axpy(tR_vu, t_dFdv, t_dFdu);

                // R_{a,u^k}
                auto tR_au = mIntegrator.a_grad_u(mTimeStep);

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

        private:
        /******************************************************************************//**
         * \brief Return solution database.
         * \return solution database
        **********************************************************************************/
        Plato::Solutions getSolution() const
        {
            Plato::Solutions tSolution(mPhysics);
            tSolution.set("State", mDisplacement);
            tSolution.set("StateDot", mVelocity);
            tSolution.set("StateDotDot", mAcceleration);
            return tSolution;
        }
    };
}

#endif
