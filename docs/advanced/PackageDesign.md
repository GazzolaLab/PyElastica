# Code Design

## Mixin and Composition

Elastica package follows Mixin and composition design patterns that may be unfamiliar to users. Here is a collection of references that introduce the package design.

### References

- [stackoverflow discussion on Mixin](https://stackoverflow.com/questions/533631/what-is-a-mixin-and-why-are-they-useful)
- [example of Mixin: python collections](https://docs.python.org/dev/library/collections.abc.html)

## Duck Typing

Elastica package uses duck typing to allow users to define their own classes and functions. Here is a `typing.Protocol` structure that is used in the package.

### Systems

``` {mermaid}
    flowchart LR
      direction RL
      subgraph Systems Protocol
        direction RL
        SLBD(SlenderBodyGeometryProtool)
        SymST["SymplecticSystem:\n• KinematicStates/Rates\n• DynamicStates/Rates"]
        style SymST text-align:left
        ExpST["ExplicitSystem:\n• States (Unused)"]
        style ExpST text-align:left
        P((position\nvelocity\nacceleration\n..)) --> SLBD
        subgraph StaticSystemType
            Surface
            Mesh
        end
        subgraph SystemType
            direction TB
            Rod
            RigidBody
        end
        SLBD --> SymST
        SystemType --> SymST
        SLBD --> ExpST
        SystemType --> ExpST
      end
      subgraph Timestepper Protocol
        direction TB
        StP["StepperProtocol\n• step(SystemCollection, time, dt)"]
        style StP text-align:left
        SymplecticStepperProtocol["SymplecticStepperProtocol\n• PositionVerlet"]
        style SymplecticStepperProtocol text-align:left
        ExpplicitStepperProtocol["ExpplicitStepperProtocol\n(Unused)"]
      end

      subgraph SystemCollection

      end
      SymST --> SystemCollection --> SymplecticStepperProtocol
      ExpST --> SystemCollection --> ExpplicitStepperProtocol
      StaticSystemType --> SystemCollection

```

### System Collection (Build memory block)

``` {mermaid}
    flowchart LR
      Sys((Systems))
      St((Stepper))
      subgraph SystemCollectionType
        direction LR
        StSys["StaticSystem:\n• Surface\n• Mesh"]
        style StSys text-align:left
        DynSys["DynamicSystem:\n• Rod\n&nbsp;&nbsp;• CosseratRod\n• RigidBody\n&nbsp;&nbsp;• Sphere\n&nbsp;&nbsp;• Cylinder"]
        style DynSys text-align:left

        BlDynSys["BlockSystemType:\n• BlockCosseratRod\n• BlockRigidBody"]
        style BlDynSys text-align:left

        F{{"Feature Group (OperatorGroup):\n• Synchronize\n• Constrain values\n• Constrain rates\n• Callback"}}
        style F text-align:left
      end
      Sys --> StSys --> F
      Sys --> DynSys -->|Finalize| BlDynSys --> St
      DynSys --> F <--> St

```

### System Collection (Features)

``` {mermaid}
    flowchart LR
      Sys((Systems))
      St((Stepper))
      subgraph SystemCollectionType
        direction LR
        StSys["StaticSystem:\n• Surface\n• Mesh"]
        style StSys text-align:left
        DynSys["DynamicSystem:\n• Rod\n&nbsp;&nbsp;• CosseratRod\n• RigidBody\n&nbsp;&nbsp;• Sphere\n&nbsp;&nbsp;• Cylinder"]
        style DynSys text-align:left

        subgraph Feature
        direction LR
        Forcing -->|add_forcing_to| Synchronize
        Constraints -->|constrain| ConstrainValues
        Constraints -->|constrain| ConstrainRates
        Contact -->|detect_contact_between| Synchronize
        Connection -->|connect| Synchronize
        Damping -->|dampen| ConstrainRates
        Callback -->|collect_diagnosis| CallbackGroup
        end
      end
      Sys --> StSys --> Feature
      Sys --> DynSys
      DynSys --> Feature <--> St

```
