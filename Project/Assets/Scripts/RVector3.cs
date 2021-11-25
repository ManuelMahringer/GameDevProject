using UnityEngine;
using System.Collections;

[System.Serializable]
public class RVector3
{
	public RVector3(int x, int y, int z)
	{
		this.x = x;
		this.y = y;
		this.z = z;
	}

	public RVector3(float x, float y, float z)
	{
		this.x = Mathf.RoundToInt(x);
		this.y = Mathf.RoundToInt(y);
		this.z = Mathf.RoundToInt(z);
	}

	public RVector3(Vector3 vector3)
	{
		this.x = Mathf.RoundToInt(vector3.x);
		this.y = Mathf.RoundToInt(vector3.y);
		this.z = Mathf.RoundToInt(vector3.z);
	}

	public Vector3 ToVector3()
	{
		return new Vector3(this.x, this.y, this.z);
	}

	public static RVector3 operator /(RVector3 a, int b)
	{
		return new RVector3(a.x / b, a.y / b, a.z / b);
	}
	public static RVector3 operator /(RVector3 a, float b)
	{
		return new RVector3(a.x / b, a.y / b, a.z / b);
	}

	public static RVector3 operator *(RVector3 a, RVector3 b)
	{
		return new RVector3(a.x * b.z, a.y * b.z, a.z * b.z);
	}
	public static RVector3 operator *(RVector3 a, float b)
	{
		return new RVector3(a.x * b, a.y * b, a.z * b);
	}

	public static implicit operator Vector3(RVector3 a)
	{
		return (Vector3)a.ToVector3();
	}

	public override string ToString()
	{
		return string.Format("RVector : [{0},{1},{2}]", x, y, z);
	}

	public int x;
	public int y;
	public int z;
}
