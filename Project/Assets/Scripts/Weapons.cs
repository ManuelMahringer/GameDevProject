using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Netcode;
using UnityEngine;


public abstract class Weapon {
    public abstract string Name { get; }
    public abstract float Damage { get; }
    public abstract float Range { get; }
}

public class AssaultRifle : Weapon {
    public override string Name => "AssaultRifle";
    public override float Damage => 30f;
    public override float Range => 300f;
}

public class Handgun : Weapon {
    public override string Name => "Handgun";
    public override float Damage => 10f;
    public override float Range => 50f;
}

public class Shovel : Weapon {
    public override string Name => "Shovel";
    public override float Damage => 30f;
    public override float Range => 1f;
}