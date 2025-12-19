# Code Design

## Mixin and Composition

Elastica package follows Mixin and composition design patterns that may be unfamiliar to users. Here is a collection of references that introduce the package design.

### References

- [stackoverflow discussion on Mixin](https://stackoverflow.com/questions/533631/what-is-a-mixin-and-why-are-they-useful)
- [example of Mixin: python collections](https://docs.python.org/dev/library/collections.abc.html)

## Structural subtyping

Elastica package uses [structural subtyping](https://peps.python.org/pep-0544/) to allow users to define their own classes and functions. Here is a `typing.Protocol` structure that is used in the package.

### Systems

``` {mermaid}
    flowchart LR
      direction RL
      subgraph Systems Protocol
        direction RL
        SymST["SymplecticSystemProtocol<br/>(Necessary to be stepped by the timestepper)"]
        style SymST text-align:left
        StaticSystemType["Static System Type"<br/>• Plane]
        SystemType["(Dynamic) System Type<br/>• CosseratRod (Rod)<br/>• Sphere (RigidBody)<br/>• Cylinder (RigidBody)"]

        SystemType --> SymST
      end

      subgraph System Collection
        SysColl["SystemCollectionProtocol"]
      end

      subgraph Timestepper Protocol
        direction LR
        SymplecticStepperProtocol["SymplecticStepperProtocol<br/>• PositionVerlet"]
        style SymplecticStepperProtocol text-align:left
      end

      SymST --> SysColl -->|Symplectic systems only| SymplecticStepperProtocol
      StaticSystemType --> SysColl
```

#### Key takeaways:

- Any object that conforms to `StaticSystemProtocol` can be added to the system collection.
    - If you want to add custom type to the system, you can use `append_allowed_types` to add it to the system collection. To add associated block support, you can use `enable_block_supports`.
- Among the systems added to the system collection, only objects that conform to `SystemProtocol` will be integrated by the timestepper.
- If block support is available for a system, they will be collected together during the `finalize` step, and passed to the timestepper.


### System Collection (Build memory block)

``` {mermaid}
    flowchart LR
      Sys((Systems))
      St((Stepper))
      subgraph SystemCollectionType
        direction LR
        StSys["StaticSystem:<br/>• Plane"]
        style StSys text-align:left
        DynSys["DynamicSystem:<br/>• CosseratRod<br/>• Sphere<br/>• Cylinder"]
        style DynSys text-align:left

        BlDynSys["BlockSystem:<br/>• MemoryBlockCosseratRod<br/>• MemoryBlockRigidBody"]
        style BlDynSys text-align:left

        F{{"Feature Group (OperatorGroup):<br/>• Synchronize<br/>• Constrain values<br/>• Constrain rates<br/>• Callback"}}
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
        StSys["StaticSystem:<br/>• Plane"]
        style StSys text-align:left
        DynSys["DynamicSystem:<br/>• Rod<br/>&nbsp;&nbsp;• CosseratRod<br/>• RigidBody<br/>&nbsp;&nbsp;• Sphere<br/>&nbsp;&nbsp;• Cylinder"]
        style DynSys text-align:left

        subgraph Feature
        direction LR
        Forcing -->|add_forcing_to| Synchronize
        Constraints -->|constrain| ConstrainValues
        Constraints -->|constrain| ConstrainRates
        Contact -->|detect_contact_between| Synchronize
        Connection -->|connect| Synchronize
        Damping -->|dampen| ConstrainRates
        Callback -->|collect_diagnostics| CallbackGroup
        end
      end
      Sys --> StSys --> Feature
      Sys --> DynSys
      DynSys --> Feature <--> St

```
