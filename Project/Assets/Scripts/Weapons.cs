using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Netcode;
using UnityEngine;


public abstract class Weapon {
    public abstract string Name { get; }
    public abstract float Damage { get; }
    public abstract float Range { get; }
    public abstract float Firerate { get; }

    public float LerpDamage(float hitDist) {
        float t = 1 - (hitDist / Range);
        return Mathf.Lerp(0, Damage, t);
    }
}


public class Handgun : Weapon {
    public override string Name => "Handgun";
    public override float Damage => 30f;
    public override float Range => 5f;
    public override float Firerate => 0.5f;
}

public class AssaultRifle : Weapon {
    public override string Name => "AssaultRifle";
    public override float Damage => 50f;
    public override float Range => 30f;
    public override float Firerate => 1f;
}

public class Shovel : Weapon {
    public override string Name => "Shovel";
    public override float Damage => 30f;
    public override float Range => 1f;
    public override float Firerate { get; }
}