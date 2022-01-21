using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Netcode;
using UnityEngine;

public enum WeaponType {
    Handgun,
    AssaultRifle,
    Shovel
}


public abstract class Weapon {
    public abstract WeaponType WeaponType { get; }
    public abstract float Damage { get; }
    public abstract float Range { get; }
    public abstract float Firerate { get; }

    public float LerpDamage(float hitDist) {
        float t = 1 - (hitDist / Range);
        return Mathf.Lerp(0, Damage, t);
    }
}


public class Handgun : Weapon {
    public override WeaponType WeaponType => WeaponType.Handgun;
    public override float Damage => 30f;
    public override float Range => 70f;
    public override float Firerate => 0.3f;
}

public class AssaultRifle : Weapon {
    public override WeaponType WeaponType => WeaponType.AssaultRifle;
    public override float Damage => 50f;
    public override float Range => 250f;
    public override float Firerate => 1f;
}

public class Shovel : Weapon {
    public override WeaponType WeaponType => WeaponType.Shovel;
    public override float Damage => 25f;
    public override float Range => 3.5f;
    public override float Firerate => 0.7f;
}