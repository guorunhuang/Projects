package assignment1.problem1;

public interface IDatabase {
  boolean isValidRecipient(String id, String name);

  void depositToAccount(String id, int amount);
}
