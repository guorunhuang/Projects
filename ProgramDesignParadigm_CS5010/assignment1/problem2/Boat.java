package assignment1.problem2;

/**
 * Represents a Boat vessel.
 */
public class Boat extends Vehicle {
  private float length;
  private Integer passengers;
  private PropulsionType propulsionType;

  public Boat(String id, Integer year, MakeModel mm, double msrp,
      float length, Integer passengers, PropulsionType type) {
    super(id, year, mm, msrp);
    this.length = length;
    this.passengers = passengers;
    this.propulsionType = type;
  }

  public float getLength() { return this.length; }
  public Integer getPassengers() { return this.passengers; }
  public PropulsionType getPropulsionType() { return this.propulsionType; }
}